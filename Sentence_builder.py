# app.py
# This script is a simple version of the ISL app.
# It shows the camera feed, detects gestures, and builds a sentence from the predicted words.

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
from grammar import grammar_correction

from config import (
    MODEL_PATH,
    ENCODER_PATH,
    CAMERA_INDEX,
    STABLE_FRAMES_THRESHOLD,
)

# ---------- Load model and label encoder ----------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_landmarks(results):
    """
    Convert the MediaPipe hand landmarks into a flat list.

    - Uses up to 2 hands.
    - Each hand has 21 points → (x, y, z) = 63 values.
    - So two hands together = 126 values.
    - If only one hand is visible, the second hand is filled with zeros.
    - If no hands are detected, returns None.
    """
    if not results.multi_hand_landmarks:
        return None

    landmark_list = []
    hand_count = len(results.multi_hand_landmarks)

    # Take at most 2 hands
    for handLms in results.multi_hand_landmarks[:2]:
        for lm in handLms.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

    # If only 1 hand, pad zeros for the second hand
    if hand_count == 1:
        landmark_list.extend([0.0] * (21 * 3))

    return landmark_list


def clean_sentence(words):
    """
    Simple formatting helper.
    Example:
        ["HELLO", "INDIAN", "SIGN", "LANGUAGE"]
        -> "Hello Indian Sign Language."
    """
    if not words:
        return ""
    s = " ".join(words)
    s = s.title()  # Capitalize each word
    if not s.endswith("."):
        s += "."
    return s


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # For smoothing predictions over multiple frames
    prediction_history = deque(maxlen=STABLE_FRAMES_THRESHOLD)

    # Words collected so far for the current sentence
    current_sentence_words = []

    # Last completed sentence
    display_sentence = ""

    # To avoid adding the same stable word again and again
    last_accepted_word = None

    print("Controls: 'e' = end sentence, 'b' = backspace, 'c' = clear, 'q' = quit")

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Run MediaPipe on the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)

            # Draw hand landmarks on the frame (just for visual feedback)
            if hand_results.multi_hand_landmarks:
                for handLms in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, handLms, mp_hands.HAND_CONNECTIONS
                    )

            current_word = "None"
            max_conf = 0.0

            # Get landmarks and make a prediction
            landmarks = extract_hand_landmarks(hand_results)

            if landmarks is not None:
                X = np.array(landmarks).reshape(1, -1)
                try:
                    probs = model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(probs))
                    current_word = label_encoder.inverse_transform([pred_idx])[0]
                    max_conf = float(probs[pred_idx])

                    # Only keep predictions that are confident enough
                    if max_conf >= 0.7:
                        prediction_history.append(current_word)
                except Exception:
                    current_word = "Error"

            # Decide a stable word based on repeated predictions
            stable_word = None
            if len(prediction_history) == STABLE_FRAMES_THRESHOLD:
                most_common_word, count = Counter(prediction_history).most_common(1)[0]
                if count >= STABLE_FRAMES_THRESHOLD:
                    stable_word = most_common_word

            # Add stable word to the sentence if it is new
            if stable_word and stable_word != last_accepted_word:
                last_accepted_word = stable_word
                current_sentence_words.append(stable_word)

            # ---------- Text overlay on the video ----------
            cv2.putText(frame, f"Word: {current_word}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            building_text = " ".join(current_sentence_words)
            cv2.putText(frame, f"Building: {building_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Sentence: {display_sentence}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # ---------- Controls box at bottom-right ----------
            controls = [
                "Controls:",
                "e : End sentence",
                "b : Backspace",
                "c : Clear",
                "q : Quit"
            ]

            box_x1 = w - 220
            box_y1 = h - 140
            box_x2 = w - 10
            box_y2 = h - 10

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            y_offset = box_y1 + 25
            for line in controls:
                cv2.putText(frame, line, (box_x1 + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y_offset += 25

            cv2.imshow("ISL Sentence-level App", frame)

            # ---------- Keyboard controls ----------
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('e'):
                # Confirm the current sentence and clear for next one
                if current_sentence_words:
                    display_sentence = grammar_correction(current_sentence_words)
                    current_sentence_words = []
                    prediction_history.clear()
                    last_accepted_word = None

            elif key == ord('b'):
                # Remove the last word from the building sentence
                if current_sentence_words:
                    current_sentence_words.pop()
                    prediction_history.clear()
                    last_accepted_word = None

            elif key == ord('c'):
                # Clear the whole sentence being built
                current_sentence_words = []
                prediction_history.clear()
                last_accepted_word = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
