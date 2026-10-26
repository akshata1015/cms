"""
Test the detector on a single image file instead of live camera.
Usage: python test_image.py path/to/photo.jpg
"""
import sys
import cv2
from crime_detector import CrimeDetector

if len(sys.argv) < 2:
    print("Usage: python test_image.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
frame = cv2.imread(image_path)
if frame is None:
    print(f"Could not read image: {image_path}")
    sys.exit(1)

detector = CrimeDetector()
annotated, fight, plate_alerts = detector.process_frame(frame)

# A close-up plate image may not contain enough vehicle context for YOLO to
# create a vehicle box, so test it directly as an ANPR image as a fallback.
if not plate_alerts:
    height, width = frame.shape[:2]
    plate_text = detector.read_plate(frame, 0, 0, width, height)
    if plate_text:
        plate_alerts = [(plate_text, plate_text in detector.stolen_plates)]

print("\n--- RESULTS ---")
print("Fight detected:", fight)
print("Plate alerts:", plate_alerts if plate_alerts else "None found")

for name, box in detector.check_wanted_faces(frame):
    print("Wanted face matched:", name)

cv2.imshow("Test Result (press any key to close)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
