import cv2
import time

net = cv2.dnn.readNetFromONNX("model.onnx")
video = cv2.VideoCapture("video.mp4")

smoothed = 0

while True:
    retval, display = video.read()
    if retval == False: break

    height, width, _ = display.shape
    new_width = int(width * 300 / height)
    new_height = 300
    frame = cv2.resize(display, (new_width, new_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.threshold(frame, 128, 1, cv2.THRESH_BINARY)[1]

    closed = cv2.morphologyEx(
        frame, 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)))
    
    contours, hierarchy = cv2.findContours(
        closed, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
    
    segments = []
    for contour in contours:
        if cv2.contourArea(contour) < 30: continue
        rect = cv2.boundingRect(contour)
        segments.append(rect)
    segments.sort(key=lambda x: x[0])

    number = ""
    outputs = []
    average = 0
    for segment in segments:
        x, y, w, h = segment
        buffer = frame[y:y + h, x:x + w]
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if w > h:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
        else:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
        buffer = cv2.copyMakeBorder(
            buffer, 
            pad_top, 
            pad_bottom, 
            pad_left, 
            pad_right, 
            cv2.BORDER_CONSTANT, 
            value=0)
        blob = cv2.dnn.blobFromImage(buffer, size=(16, 16))
        net.setInput(blob)
        start_time = time.perf_counter()
        result = net.forward()
        end_time = time.perf_counter()
        average += (end_time - start_time) * 1000000
        min, max, min_loc, max_loc = cv2.minMaxLoc(result)
        number += str(max_loc[0])
        x1 = x / new_width
        y1 = y / new_height
        x2 = (x + w) / new_width
        y2 = (y + h) / new_height

        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        cv2.rectangle(display, p1, p2, (255, 0, 0), 2)

    if len(segments) > 0:
        average /= len(segments)
        smoothed = 0.99 * smoothed + 0.01 * average

    cv2.putText(
        display, 
        text=f"Detected: {number}", 
        org=(20, 50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, 
        color=(255, 255, 255), 
        thickness=2)
    
    cv2.putText(
        display, 
        text=f"Inference time: {int(smoothed)} us", 
        org=(20, 100), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, 
        color=(255, 255, 255), 
        thickness=2)

    cv2.imshow("Infer", display)
    cv2.waitKey(20)

video.release()