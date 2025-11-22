import cv2

for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera detected at index {i}")
        break
    cap.release()

if not cap.isOpened():
    print("No camera found!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
