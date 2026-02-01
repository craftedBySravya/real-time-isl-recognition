# app_chat.py
# Main real-time chat view.
# Left: camera + sign → sentence.
# Right: chat between signer and non-signer.

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

from config import (
    MODEL_PATH,
    ENCODER_PATH,
    CAMERA_INDEX,
    STABLE_FRAMES_THRESHOLD,
)

from grammar import grammar_correction
from emotion import classify_face_emotion, estimate_emotion
from speech import speak_text
from translate import translate_in_background

# -----------------------------
#  Settings for gesture filter
# -----------------------------
MIN_CONFIDENCE = 0.75          # minimum confidence to trust a prediction
MIN_MARGIN = 0.05              # top1 - top2 margin
IGNORE_LABELS = {"END"}        # labels that should not be added as words
WORD_COOLDOWN_FRAMES = 20      # small delay after adding a word


# -----------------------
# Load model + encoder
# -----------------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def extract_hand_landmarks(results):
    """
    Convert hand landmarks into a single list.
    - Up to 2 hands.
    - If only 1 hand is visible, second hand is padded with zeros.
    - If no hands, return None.
    """
    if not results.multi_hand_landmarks:
        return None

    data = []
    hand_count = len(results.multi_hand_landmarks)

    # first 2 hands only
    for handLms in results.multi_hand_landmarks[:2]:
        for lm in handLms.landmark:
            data.extend([lm.x, lm.y, lm.z])

    if hand_count == 1:
        # pad for second hand
        data.extend([0.0] * (21 * 3))

    return data


def main():

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # for smoothing words over frames
    prediction_history = deque(maxlen=STABLE_FRAMES_THRESHOLD)

    # words from current signed sentence
    current_sentence_words = []
    last_accepted_word = None

    # last finished sentence + emotion
    display_sentence = ""
    display_emotion = ""

    # chat list: (sender, text, emotion)
    chat_history = []

    # typing support for hearing person
    typing_mode = False
    typing_buffer = ""

    # frame-wise emotions during one signed sentence
    sentence_emotion_labels = []

    # cooldown counter after accepting a word
    cooldown_frames = 0

    print("Controls:")
    print(" e=end sentence | b=back | c=clear | t=type | q=quit\n")

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        while True:

            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # --------------------------------
            # Split screen: left = video area
            # --------------------------------
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            left_w = w // 2

            scale = left_w / float(w)
            vid_h = int(h * scale)

            resized_frame = cv2.resize(frame, (left_w, vid_h))
            y0 = (h - vid_h) // 2
            y1 = y0 + vid_h

            canvas[y0:y1, 0:left_w] = resized_frame
            disp = canvas

            # --------------------------------
            # Run Mediapipe for hands + face
            # --------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            # draw hands on left side
            if hand_results.multi_hand_landmarks:
                region = disp[y0:y1, 0:left_w]
                for handLms in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        region,
                        handLms,
                        mp_hands.HAND_CONNECTIONS
                    )
                disp[y0:y1, 0:left_w] = region

            # get rough face emotion (frame level)
            current_face_emotion = None
            if face_results.multi_face_landmarks:
                face_lms = face_results.multi_face_landmarks[0]
                current_face_emotion = classify_face_emotion(face_lms)

            # --------------------------------
            # Gesture prediction with stability
            # --------------------------------
            current_word = "None"
            stable_word = None

            # skip prediction during cooldown
            if cooldown_frames > 0:
                cooldown_frames -= 1
            else:
                landmarks = extract_hand_landmarks(hand_results)

                if landmarks is not None:
                    X = np.array(landmarks).reshape(1, -1)

                    try:
                        probs = model.predict_proba(X)[0]

                        sorted_idx = np.argsort(probs)
                        top1 = sorted_idx[-1]
                        top2 = sorted_idx[-2]

                        conf1 = probs[top1]
                        conf2 = probs[top2]
                        margin = conf1 - conf2

                        word1 = label_encoder.inverse_transform([top1])[0]
                        current_word = word1

                        # keep only strong predictions
                        if conf1 >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
                            prediction_history.append(word1)

                    except Exception:
                        current_word = "Error"

                # pick the most common word from recent frames
                if len(prediction_history) == STABLE_FRAMES_THRESHOLD:
                    most, count = Counter(prediction_history).most_common(1)[0]
                    if count >= STABLE_FRAMES_THRESHOLD:
                        stable_word = most
                        prediction_history.clear()

            # --------------------------------
            # If we got a stable word, add it
            # --------------------------------
            if stable_word and stable_word != last_accepted_word:

                if stable_word not in IGNORE_LABELS:
                    current_sentence_words.append(stable_word)
                    last_accepted_word = stable_word

                    if current_face_emotion is not None:
                        sentence_emotion_labels.append(current_face_emotion)

                    cooldown_frames = WORD_COOLDOWN_FRAMES
                    prediction_history.clear()

            # --------------------------------
            # Left side text (video overlay)
            # --------------------------------
            base_y = y0 + 25

            cv2.putText(disp, f"Word: {current_word}", (10, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            building = " ".join(current_sentence_words)
            cv2.putText(disp, f"Building: {building}", (10, base_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            last_msg = f"Last: {display_sentence}"
            if display_sentence and display_emotion:
                last_msg += f" ({display_emotion})"

            cv2.putText(disp, last_msg, (10, base_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # controls at bottom of left side
            cv2.rectangle(disp, (0, h - 65), (left_w, h), (0, 0, 0), -1)
            lines = [
                "e=end | t=type | b=back | c=clear | q=quit",
                "Typing: Enter=send | Esc=cancel | Backspace"
            ]
            cy = h - 42
            for line in lines:
                cv2.putText(disp, line, (10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 255, 230), 1)
                cy += 20

            # --------------------------------
            # Right side: chat area
            # --------------------------------
            px1 = left_w
            px2 = w
            py1 = 0
            py2 = h

            cv2.rectangle(disp, (px1, py1), (px2, py2), (25, 25, 25), -1)
            cv2.putText(disp, "Chat", (px1 + 15, py1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(disp, "(Signer & Non-signer)", (px1 + 15, py1 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # show recent messages
            y = py1 + 95
            for sender, text, emo in chat_history[-6:]:
                tag = "Signer:" if sender == "Signer" else "Non-signer:"
                color = (0, 255, 255) if sender == "Signer" else (200, 200, 255)

                cv2.putText(disp, tag, (px1 + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                y += 25

                msg = text
                if sender == "Signer" and emo:
                    msg += f" ({emo})"

                cv2.putText(disp, msg, (px1 + 30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 35

            # typing box at bottom of chat panel
            box_y = py2 - 50
            cv2.rectangle(disp, (px1 + 10, box_y), (px2 - 10, py2 - 10),
                          (50, 50, 50), -1)

            prompt = "Type (Enter=send):" if typing_mode else "Press 't' to type reply"
            cv2.putText(disp, prompt, (px1 + 15, box_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if typing_mode:
                cv2.putText(disp, typing_buffer + "|",
                            (px1 + 15, py2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)

            cv2.imshow("ISL Real-time Interaction (Chat View)", disp)

            # --------------------------------
            # Keyboard handling
            # --------------------------------
            key = cv2.waitKey(1) & 0xFF

            # typing mode: non-signer entering text
            if typing_mode:
                if key in (13, 10):  # Enter
                    text_out = typing_buffer.strip()
                    if text_out:
                        chat_history.append(("Non-signer", text_out, None))
                        speak_text(text_out)
                    typing_buffer = ""
                    typing_mode = False

                elif key == 27:  # ESC
                    typing_buffer = ""
                    typing_mode = False

                elif key in (8, 127):  # Backspace
                    typing_buffer = typing_buffer[:-1]

                elif key != 255:
                    ch = chr(key)
                    if ch.isprintable():
                        typing_buffer += ch

                # when typing, ignore gesture controls
                continue

            # normal mode
            if key == ord('q'):
                break

            elif key == ord('t'):
                typing_mode = True
                typing_buffer = ""

            elif key == ord('b'):
                # remove last signed word
                if current_sentence_words:
                    current_sentence_words.pop()
                if sentence_emotion_labels:
                    sentence_emotion_labels.pop()
                prediction_history.clear()
                last_accepted_word = None

            elif key == ord('c'):
                # clear current signed sentence
                current_sentence_words = []
                sentence_emotion_labels = []
                prediction_history.clear()
                last_accepted_word = None

            elif key == ord('e'):
                # finish signed sentence
                if current_sentence_words:
                    final_sentence = grammar_correction(current_sentence_words)
                    emo = estimate_emotion(sentence_emotion_labels)

                    display_sentence = final_sentence
                    display_emotion = emo

                    chat_history.append(("Signer", final_sentence, emo))

                    # show translations in terminal
                    translate_in_background(final_sentence)

                    # reset for next sentence
                    current_sentence_words = []
                    sentence_emotion_labels = []
                    prediction_history.clear()
                    last_accepted_word = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
