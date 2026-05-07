import 'package:flutter/material.dart';

class RectPainter extends CustomPainter {
  RectPainter({required this.rects}) : super(repaint: rects);

  final ValueNotifier<List<double>> rects;
  final _paint = Paint()
    ..strokeWidth = 2
    ..color = Colors.blue
    ..style = PaintingStyle.stroke;

  @override
  void paint(Canvas canvas, Size size) {
    if (rects.value.isEmpty) return;
    for (int i = 0; i < rects.value.length; i += 4) {
      canvas.drawRect(
        Rect.fromPoints(
          Offset(
            rects.value[i] * size.width,
            rects.value[i + 1] * size.height,
          ),
          Offset(
            rects.value[i + 2] * size.width,
            rects.value[i + 3] * size.height,
          ),
        ),
        _paint,
      );
    }
  }

  @override
  bool shouldRepaint(RectPainter oldDelegate) => false;
}
