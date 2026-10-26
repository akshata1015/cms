import cv2

for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
    for index in range(4):
        cap = cv2.VideoCapture(index, backend)
        ok = cap.isOpened()
        ret, frame = cap.read() if ok else (False, None)
        print(f"{backend_name} index={index} opened={ok} frame={ret}")
        cap.release()