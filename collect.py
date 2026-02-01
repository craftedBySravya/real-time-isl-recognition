# collect.py
# This script is used to record gesture samples and save them into CSV files.

import cv2
import mediapipe as mp
import csv
import os
import time

from config import DATASET_DIR, SAMPLES_PER_GESTURE, CAMERA_INDEX

# -------- MediaPipe Setup --------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Make sure dataset folder exists
os.makedirs(DATASET_DIR, exist_ok=True)


def extract_landmarks(results):
    """
    Gets hand landmarks for up to 2 hands.
    Each hand has 21 points → (x, y, z) = 63 values.
    For consistency, always return 126 values.
    If only one hand is seen, the second hand is filled with zeros.
    If no hands are detected, return None.
    """
    if not results.multi_hand_landmarks:
        return None

    landmark_list = []
    hand_count = len(results.multi_hand_landmarks)

    # Take only the first 2 hands (if more are seen)
    for handLms in results.multi_hand_landmarks[:2]:
        for lm in handLms.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

    # If one hand detected → pad the rest with zeros
    if hand_count == 1:
        landmark_list.extend([0.0] * (21 * 3))

    return landmark_list


def main():
    print("\n=== GESTURE DATA COLLECTION ===")

    gesture_label = input("Enter gesture name (e.g. HELLO, WATER, HELP): ").strip()
    if not gesture_label:
        print("No gesture entered, exiting.")
        return

    # Use uppercase for consistency
    gesture_label = gesture_label.upper()

    # CSV file where this gesture's data will be saved
    csv_path = os.path.join(DATASET_DIR, f"{gesture_label}.csv")
    print(f"\nGesture: {gesture_label}")
    print(f"Saving data to: {csv_path}")

    print("\nInstructions:")
    print("  - Press 's' to start recording.")
    print(f"  - Keep the gesture steady while {SAMPLES_PER_GESTURE} samples are recorded.")
    print("  - Press 'q' to quit anytime.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Open CSV in append mode so you can record more batches later
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            recording = False
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera error.")
                    break

                frame = cv2.flip(frame, 1)

                # Process the frame for hand landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Draw hand landmarks (just for visual help)
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Status text on screen
                cv2.putText(frame, f"Gesture: {gesture_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                status_text = "Press 's' to start" if not recording else f"Recording... {count}/{SAMPLES_PER_GESTURE}"
                cv2.putText(frame, status_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("Collect Gesture Data", frame)

                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key == ord('q'):
                    print("Exiting.")
                    break

                # Start recording
                if key == ord('s') and not recording:
                    print("\nStarting in 1 second... Hold the gesture steady.")
                    time.sleep(1)
                    recording = True

                # While recording, collect the samples
                if recording and count < SAMPLES_PER_GESTURE:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    landmarks = extract_landmarks(results)

                    if landmarks is not None:
                        # Add the gesture label at the end
                        row = landmarks + [gesture_label]
                        writer.writerow(row)
                        count += 1

                    # Update recording text
                    cv2.putText(frame, f"Recording... {count}/{SAMPLES_PER_GESTURE}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Gesture: {gesture_label}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw landmarks again
                    if results.multi_hand_landmarks:
                        for handLms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    cv2.imshow("Collect Gesture Data", frame)
                    cv2.waitKey(1)

                    # Done collecting
                    if count >= SAMPLES_PER_GESTURE:
                        print(f"\nFinished collecting {SAMPLES_PER_GESTURE} samples for {gesture_label}")
                        recording = False
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
