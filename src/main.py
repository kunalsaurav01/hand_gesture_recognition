import cv2
import mediapipe as mp
from gesture_logic import classify_gesture

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip for selfie-view display and convert to RGB
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = "No Hand Detected"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                gesture_text = classify_gesture(hand_landmarks)
                # Draw keypoints and skeleton
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
        # Overlay gesture label
        cv2.putText(image, f"Gesture: {gesture_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show
        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
            break
cap.release()
cv2.destroyAllWindows()
