import cv2

cap = cv2.VideoCapture("rtsp://admin@172.20.208.1:554/cam/realmonitor?channel=2&subtype=0")

if not cap.isOpened():
    print('VideoCapture not opened')
    exit(-1)

while True:
    ret, frame = cap.read()

    if not ret:
        print('frame read failed')
        break

    cv2.imshow('image', frame)

    if cv2.waitKey(1)&0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
