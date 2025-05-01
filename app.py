import cv2
import random
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


model = tf.keras.models.load_model("sign_language_mobilenet.h5")


class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Grad-CAM
replace2linear = ReplaceToLinear()
gradcam = Gradcam(model, model_modifier=replace2linear)

# Start webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(2)  # Allow camera warm-up


mode = 'normal'  # 'dim', 'occlusion', 'rotate' also valid
gradcam_enabled = False

def apply_perturbation(frame, mode):
    if mode == 'dim':
        return cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
    elif mode == 'occlusion':
        h, w, _ = frame.shape
        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        frame = frame.copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), -1)
        return frame
    elif mode == 'rotate':
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
        return cv2.warpAffine(frame, M, (w, h))
    else:
        return frame

def generate_gradcam(input_image, class_idx):
    input_tensor = tf.convert_to_tensor(input_image)
    score = CategoricalScore(class_idx)
    cam = gradcam(score, input_tensor)
    heatmap = np.uint8(255 * cam[0])
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    frame = cv2.flip(frame, 1)
    frame = apply_perturbation(frame, mode)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                input_img = cv2.resize(hand_img, (224, 224))
                input_img_norm = np.expand_dims(input_img / 255.0, axis=0)
                prediction = model.predict(input_img_norm, verbose=0)
                idx = np.argmax(prediction)
                predicted_class = class_names[idx]
                confidence = np.max(prediction) * 100

                # Apply confidence threshold
                CONFIDENCE_THRESHOLD = 60
                if confidence < CONFIDENCE_THRESHOLD:
                    predicted_class = "Uncertain"
                    label = "Low Confidence"
                else:
                    label = f"{predicted_class} ({confidence:.2f}%)"

                # Log results
                with open("robustness_logs.csv", "a") as f:
                    f.write(f"{mode},{predicted_class},{confidence:.2f}\n")

                # GradCAM visualization
                if gradcam_enabled:
                    heatmap = generate_gradcam(input_img_norm, idx)
                    heatmap = cv2.resize(heatmap, (224, 224))
                    overlay = cv2.addWeighted(input_img, 0.6, heatmap, 0.4, 0)
                    frame[y_min:y_max, x_min:x_max] = cv2.resize(overlay, (x_max - x_min, y_max - y_min))
                    cv2.imwrite(f"gradcam_{mode}_{predicted_class}.jpg", overlay)

                # Draw label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_x = x_min
                label_y = y_min - 10 if y_min - 10 > 10 else y_min + 20
                cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5),
                              (label_x + label_size[0], label_y + 5), (0, 255, 0), -1)
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display status
    status = f"Mode: {mode.upper()} | GradCAM: {'ON' if gradcam_enabled else 'OFF'}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = 'normal'
    elif key == ord('2'):
        mode = 'dim'
    elif key == ord('3'):
        mode = 'occlusion'
    elif key == ord('4'):
        mode = 'rotate'
    elif key == ord('g'):
        gradcam_enabled = not gradcam_enabled
        print(f"GradCAM Enabled: {gradcam_enabled}")

cap.release()
cv2.destroyAllWindows()
