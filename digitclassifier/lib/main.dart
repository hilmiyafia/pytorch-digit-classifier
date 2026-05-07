import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'rect_painter.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Digit Classifier',
      theme: ThemeData(colorScheme: .fromSeed(seedColor: Colors.deepPurple)),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with WidgetsBindingObserver {
  CameraController? _cameraController;

  var _lastRun = 0;
  var _working = false;
  var _loaded = false;
  var _onDebug = false;

  final _bytes = ValueNotifier<Uint8List?>(null);

  final _message = ValueNotifier<String>('');
  final _rects = ValueNotifier<List<double>>([]);
  late final cv.Mat _kernel;
  late final cv.Net _net;

  var _smoothed = 0.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    initCamera();

    _kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 17));
    () async {
      _net = cv.Net.fromOnnx(await loadAssetToFile("assets/model.onnx"));
      _loaded = true;
    }();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null) return;
    if (_cameraController!.value.isInitialized == false) return;
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _kernel.dispose();
    super.dispose();
  }

  Future<void> initCamera() async {
    final cameras = await availableCameras();
    final index = cameras.indexWhere((camera) => camera.lensDirection == .back);
    if (index < 0) return;
    _cameraController = CameraController(
      cameras[index],
      .medium,
      enableAudio: false,
      imageFormatGroup: .yuv420,
    );
    try {
      await _cameraController!.initialize();
      await _cameraController!.startImageStream(processImage);
    } catch (e) {
      _message.value = e.toString();
    }
  }

  Future<String> loadAssetToFile(String assetPath) async {
    final directory = await getApplicationSupportDirectory();
    final fileName = assetPath.split('/').last;
    final localPath = "${directory.path}/$fileName";
    final file = File(localPath);
    if (await file.exists() == false) {
      final data = await rootBundle.load(assetPath);
      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      await file.writeAsBytes(bytes, flush: true);
    }
    return localPath;
  }

  void processImage(CameraImage image) async {
    if (_loaded == false) return;
    if (mounted == false) return;
    final timeNow = DateTime.now().millisecondsSinceEpoch;
    if (_working || timeNow - _lastRun < 32) return;
    final cameras = await availableCameras();
    int index = cameras.indexWhere((camera) => camera.lensDirection == .back);
    _working = true;
    _rects.value = detectRects(image, cameras[index].sensorOrientation);
    _working = false;
    _lastRun = DateTime.now().millisecondsSinceEpoch;
    if (mounted == false) return;
  }

  cv.Mat imageToMat(CameraImage image) {
    final ySize = image.planes[0].bytesPerRow * image.height;
    final uSize = image.planes[1].bytesPerRow * image.height;
    final vSize = image.planes[2].bytesPerRow * image.height;
    final bytes = Uint8List(ySize + uSize + vSize);
    bytes.setAll(0, image.planes[0].bytes);
    bytes.setAll(ySize, image.planes[1].bytes);
    bytes.setAll(ySize + uSize, image.planes[2].bytes);
    final buffer = cv.Mat.fromList(
      image.height + image.height ~/ 2,
      image.planes[0].bytesPerRow,
      cv.MatType.CV_8UC1,
      bytes,
    );
    final frame = cv.cvtColor(buffer, cv.COLOR_YUV2BGR_NV12);
    final cropped = frame.region(cv.Rect(0, 0, image.width, image.height));
    final result = cropped.clone();
    buffer.dispose();
    frame.dispose();
    cropped.dispose();
    return result;
  }

  List<double> detectRects(CameraImage image, int rotation) {
    var frame = imageToMat(image);
    if (rotation == 90) {
      final rotated = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE);
      frame.dispose();
      frame = rotated;
    }
    final newWidth = frame.width * 300 ~/ frame.height;
    final newHeight = 300;
    final resized = cv.resize(frame, (newWidth, newHeight));
    final gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
    final threshold = cv.threshold(gray, 224, 1, cv.THRESH_BINARY).$2;
    final closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, _kernel);
    final (contours, hierarchy) = cv.findContours(
      closed,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    if (_onDebug) {
      final scale = cv.Mat.fromScalar(
        threshold.rows,
        threshold.cols,
        cv.MatType.CV_8UC1,
        cv.Scalar(255),
      );
      final debug = cv.multiply(closed, scale);
      final buffer = cv.resize(debug, (frame.width, frame.height));
      _bytes.value = cv.imencode(".jpg", buffer).$2;
      scale.dispose();
      debug.dispose();
      buffer.dispose();
    }
    final rects = <cv.Rect>[];
    for (final contour in contours) {
      if (cv.contourArea(contour) < 30) continue;
      rects.add(cv.boundingRect(contour));
    }
    contours.dispose();
    hierarchy.dispose();
    rects.sort((a, b) => a.x.compareTo(b.x));
    var number = '';
    final outputs = <double>[];
    var average = 0.0;
    for (final rect in rects) {
      final buffer = threshold.region(rect);
      var top = 0, bottom = 0, left = 0, right = 0;
      if (buffer.width > buffer.height) {
        top = (buffer.width - buffer.height) ~/ 2;
        bottom = buffer.width - buffer.height - top;
      } else {
        left = (buffer.height - buffer.width) ~/ 2;
        right = buffer.height - buffer.width - left;
      }
      final padded = cv.copyMakeBorder(
        buffer,
        top,
        bottom,
        left,
        right,
        cv.BORDER_CONSTANT,
      );
      final blob = cv.blobFromImage(padded, size: (16, 16));
      _net.setInput(blob);
      final timer = Stopwatch()..start();
      final result = _net.forward();
      timer.stop();
      average += timer.elapsedMicroseconds;
      final (min, max, minLoc, maxLoc) = cv.minMaxLoc(result);
      number += maxLoc.x.toString();
      outputs.add(rect.x / newWidth);
      outputs.add(rect.y / newHeight);
      outputs.add(rect.right / newWidth);
      outputs.add(rect.bottom / newHeight);
      buffer.dispose();
      padded.dispose();
      blob.dispose();
      result.dispose();
    }
    if (rects.isNotEmpty) {
    average = average / rects.length;
    _smoothed = _smoothed * 0.99 + average * 0.01;
    }
    _message.value = 'Detected: $number\nInference time: ${_smoothed.toStringAsFixed(2)} us';
    frame.dispose();
    resized.dispose();
    gray.dispose();
    closed.dispose();
    threshold.dispose();
    return outputs;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text('Digit Classifier'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Center(
          child: Column(
            mainAxisAlignment: .center,
            children: [
              Checkbox(
                value: _onDebug,
                onChanged: (value) => setState(() {
                  _onDebug = value ?? false;
                }),
              ),
              () {
                if (_cameraController == null) return SizedBox.shrink();
                final ratio = 1 / _cameraController!.value.aspectRatio;
                return SizedBox(
                  height: 500,
                  child: AspectRatio(
                    aspectRatio: ratio,
                    child: Stack(
                      children: [
                        _onDebug && _bytes.value != null
                            ? ValueListenableBuilder(
                                valueListenable: _bytes,
                                builder: (context, debug, child) =>
                                    Image.memory(
                                      _bytes.value!,
                                      gaplessPlayback: true,
                                    ),
                              )
                            : CameraPreview(_cameraController!),
                        AspectRatio(
                          aspectRatio: ratio,
                          child: CustomPaint(
                            painter: RectPainter(rects: _rects),
                          ),
                        ),
                        ValueListenableBuilder(
                          valueListenable: _message,
                          builder: (context, value, child) {
                            return Text(
                              value,
                              style: const TextStyle(
                                color: Colors.white,
                                backgroundColor: Colors.black,
                                fontSize: 20,
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                );
              }(),
            ],
          ),
        ),
      ),
    );
  }
}
