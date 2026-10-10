import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc)

DATASET = "dataset/own"
os.makedirs("results", exist_ok=True)
mp_face_mesh = mp.solutions.face_mesh


def get_features(img, face_mesh):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    pts = np.array([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark])
    pts -= pts.mean(axis=0)
    pts /= (pts.max(axis=0) - pts.min(axis=0)).max()
    return pts.flatten()


# ---------- 1. READ ALL PHOTOS, PERSON NAME = START OF FILE NAME ----------
cache = "results/features_people.npz"      # delete this file if you add new photos
if os.path.exists(cache):
    d = np.load(cache)
    X, y = d["X"], d["y"]
    print("Loaded saved features")
else:
    X, y = [], []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               min_detection_confidence=0.3) as fm:
        for root, _, files in os.walk(DATASET):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")) or "_" not in f:
                    continue
                person = f.split("_")[0].strip().title()
                img = cv2.imread(os.path.join(root, f))
                if img is None:
                    continue
                feat = get_features(img, fm)
                if feat is not None:
                    X.append(feat)
                    y.append(person)
    X, y = np.array(X), np.array(y)
    np.savez(cache, X=X, y=y)

people_found, counts = np.unique(y, return_counts=True)
print("\nPhotos per person:")
for p, c in zip(people_found, counts):
    print(f"  {p}: {c}")
if len(people_found) < 2:
    sys.exit("Need photos from at least 2 people (check the file names).")

# ---------- 2. TRAIN AND TEST ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)
pred = model.predict(X_te)
people = list(model.classes_)
true_idx = np.array([people.index(t) for t in y_te])

# ---------- 3. IDENTIFICATION (1 to many) ----------
print("\n===== IDENTIFICATION (Who is this person?) =====")
print(f"Train photos: {len(X_tr)} | Test photos: {len(X_te)}")
print(f"Accuracy : {accuracy_score(y_te, pred):.4f}")
print(f"Precision: {precision_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
print(f"Recall   : {recall_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
print(f"F1-score : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}  (macro)")
print("\n", classification_report(y_te, pred, zero_division=0))

ranking = np.argsort(-proba, axis=1)
cmc = [np.mean([true_idx[i] in ranking[i, :k] for i in range(len(y_te))])
       for k in range(1, len(people) + 1)]
print("CMC (rank-k identification rate):")
for k, v in enumerate(cmc, 1):
    print(f"  Rank-{k}: {v:.4f}")

# ---------- 4. AUTHENTICATION (1 to 1) ----------
scores, labels = [], []
for i in range(len(y_te)):
    for j in range(len(people)):
        scores.append(proba[i, j])                 # "how sure is the model this is person j"
        labels.append(1 if j == true_idx[i] else 0)
fpr, tpr, _ = roc_curve(labels, scores)
fnr = 1 - tpr
k_eer = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[k_eer] + fnr[k_eer]) / 2
print("\n===== AUTHENTICATION (Is this really person X?) =====")
print(f"AUC: {auc(fpr, tpr):.4f}")
print(f"EER: {eer:.4f}  (lower is better)")

# ---------- 5. GRAPHS ----------
fig1 = plt.figure(figsize=(6, 5))
plt.plot(range(1, len(cmc) + 1), cmc, marker="o")
plt.xticks(range(1, len(cmc) + 1))
plt.ylim(0, 1.05)
plt.xlabel("Rank")
plt.ylabel("Identification rate")
plt.title("CMC Curve")
plt.grid(True)
fig1.savefig("results/cmc_curve.png", dpi=150)

fig2 = plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.2f}, EER={eer:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Accept Rate")
plt.ylabel("True Accept Rate")
plt.title("Authentication ROC")
plt.legend()
fig2.savefig("results/authentication_roc.png", dpi=150)

cm = confusion_matrix(y_te, pred, labels=people)
fig3, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(people)))
ax.set_xticklabels(people, rotation=45)
ax.set_yticks(range(len(people)))
ax.set_yticklabels(people)
for i in range(len(people)):
    for j in range(len(people)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
ax.set_xlabel("Predicted person")
ax.set_ylabel("Actual person")
ax.set_title("Identification Confusion Matrix")
fig3.tight_layout()
fig3.savefig("results/identification_confusion_matrix.png", dpi=150)

print("\nGraphs saved in the 'results' folder")
plt.show()