import os
import sys
import random
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)

# ---------- SETTINGS ----------
name = sys.argv[1] if len(sys.argv) > 1 else "fer2013"   # fer2013 or own
DATASET = os.path.join("dataset", name)
CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]
MAX_PER_CLASS = 400 if name == "fer2013" else None        # FER is huge, so we use a part of it
if len(sys.argv) > 2:
    MAX_PER_CLASS = int(sys.argv[2])

os.makedirs("results", exist_ok=True)
mp_face_mesh = mp.solutions.face_mesh


def collect_images():
    items = {c: [] for c in CLASSES}
    for root, _, files in os.walk(DATASET):
        label = os.path.basename(root).lower()
        if label in items:
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    items[label].append(os.path.join(root, f))
    random.seed(42)
    paths, labels = [], []
    for c in CLASSES:
        files = sorted(items[c])
        random.shuffle(files)
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]
        print(f"{c}: using {len(files)} images")
        paths += files
        labels += [c] * len(files)
    return paths, labels


def get_features(img, face_mesh):
    h, w = img.shape[:2]
    if max(h, w) < 192:                      # FER photos are tiny, so make them bigger
        s = 192 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    pts = np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark])
    pts -= pts.mean(axis=0)                                   # centre the face
    pts /= (pts.max(axis=0) - pts.min(axis=0)).max()          # same size for every face
    return pts.flatten()


# ---------- 1. GET THE 468 POINTS FOR EVERY PHOTO ----------
cache = f"results/features_{name}_{MAX_PER_CLASS}.npz"
if os.path.exists(cache):
    d = np.load(cache)
    X, y = d["X"], d["y"]
    print("Loaded saved features from", cache)
else:
    paths, labels = collect_images()
    if not paths:
        sys.exit(f"No photos found in {DATASET}. Check the folder names.")
    X, y = [], []
    total = {c: 0 for c in CLASSES}
    found = {c: 0 for c in CLASSES}
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               min_detection_confidence=0.3) as fm:
        for i, (p, l) in enumerate(zip(paths, labels), 1):
            total[l] += 1
            img = cv2.imread(p)
            if img is None:
                continue
            f = get_features(img, fm)
            if f is not None:
                X.append(f)
                y.append(l)
                found[l] += 1
            if i % 100 == 0:
                print(f"  processed {i}/{len(paths)}")
    X, y = np.array(X), np.array(y)
    np.savez(cache, X=X, y=y)
    print("\n----- FACE DETECTION RATE -----")
    for c in CLASSES:
        print(f"{c:9s}: {found[c]}/{total[c]} faces detected")

if len(X) < 10:
    sys.exit("Too few faces detected to train a model.")

# ---------- 2. TRAIN AND TEST ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
model.fit(X_tr, y_tr)
pred = model.predict(X_te)
proba = model.predict_proba(X_te)
classes = list(model.classes_)

# ---------- 3. NUMBERS FOR SIR ----------
print(f"\n===== RESULTS on {name} =====")
print(f"Train photos: {len(X_tr)} | Test photos: {len(X_te)}")
print(f"Accuracy : {accuracy_score(y_te, pred):.4f}")
print(f"Precision: {precision_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
print(f"Recall   : {recall_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
print(f"F1-score : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
y_bin = label_binarize(y_te, classes=classes)
print(f"AUC      : {roc_auc_score(y_bin, proba, average='macro'):.4f}  (macro, one-vs-rest)")
print("\n", classification_report(y_te, pred, zero_division=0))

# ---------- 4. GRAPHS ----------
cm = confusion_matrix(y_te, pred, labels=classes)
fig1, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix ({name})")
fig1.tight_layout()
fig1.savefig(f"results/confusion_matrix_{name}.png", dpi=150)

fig2 = plt.figure(figsize=(6, 5))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
    plt.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({name})")
plt.legend()
fig2.tight_layout()
fig2.savefig(f"results/roc_curve_{name}.png", dpi=150)

print("Graphs saved in the 'results' folder")
plt.show()