import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

dataset_path = "dataset/Arkopriya"

images = os.listdir(dataset_path)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
) as face_mesh:

    for image_name in images:
        image_path = os.path.join(dataset_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION
                )

        cv2.imshow("PrepGenie - Face Landmarks", image)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

cv2.destroyAllWindows()