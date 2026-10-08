import cv2
import os

BASE = "dataset/own"
PERSON = "Isha"   # teammates change this to their own name

keys = {
    ord("h"): "happy",
    ord("s"): "sad",
    ord("a"): "angry",
    ord("n"): "neutral",
    ord("u"): "surprise",
}

for exp in keys.values():
    os.makedirs(f"{BASE}/{exp}", exist_ok=True)

counts = {exp: len(os.listdir(f"{BASE}/{exp}")) for exp in keys.values()}

cap = cv2.VideoCapture(0)
print("h = happy | s = sad | a = angry | n = neutral | u = surprise | q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    show = frame.copy()
    text = "  ".join(f"{e[:3]}:{c}" for e, c in counts.items())
    cv2.putText(show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Capture (h s a n u, q = quit)", show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in keys:
        exp = keys[key]
        counts[exp] += 1
        cv2.imwrite(f"{BASE}/{exp}/{PERSON}_{counts[exp]}.jpg", frame)
        print("Saved", exp, counts[exp])

cap.release()
cv2.destroyAllWindows()