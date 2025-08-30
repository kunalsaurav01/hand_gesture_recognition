# gesture_recognition.py
"""
Real-time static hand gesture recognition using MediaPipe + OpenCV.
Recognizes: Open Palm, Fist, Peace Sign (V), Thumbs Up.

Press 'q' to quit.
"""
import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmarks of interest: tips and pip/mcp indices (MediaPipe indexing)
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def finger_is_up(hand_landmarks, finger_tip_id, finger_pip_id, image_height):
    """
    Return True if finger is 'up' i.e., tip is above pip (y smaller since origin at top-left).
    Works for index -> pinky.
    """
    tip_y = hand_landmarks.landmark[finger_tip_id].y * image_height
    pip_y = hand_landmarks.landmark[finger_pip_id].y * image_height
    return tip_y < pip_y  # smaller y means above

def thumb_is_extended(hand_landmarks, handedness_str, image_width):
    """
    Determine if thumb is extended. For thumb we compare x values (horizontal direction),
    but also ensure it's not only horizontally extended but away from palm center.
    Uses handedness: "Left" or "Right" from MediaPipe.
    """
    # indices
    tip = hand_landmarks.landmark[4]
    ip = hand_landmarks.landmark[3]
    mcp = hand_landmarks.landmark[2]
    # convert to pixel coords x
    tip_x = tip.x * image_width
    ip_x = ip.x * image_width
    mcp_x = mcp.x * image_width

    # For Right hand, extended thumb usually has tip_x > mcp_x (to the right) when palm faces camera
    # For Left hand, it's reversed.
    if handedness_str == "Right":
        return tip_x > mcp_x + 15  # 15 px margin to avoid noisy triggers
    else:
        return tip_x < mcp_x - 15

def thumb_points_up(hand_landmarks, image_height):
    """Check thumb vertical direction: tip above wrist (y smaller)"""
    tip_y = hand_landmarks.landmark[4].y * image_height
    wrist_y = hand_landmarks.landmark[0].y * image_height
    return tip_y < wrist_y  # pointing up if tip has smaller y than wrist

def classify_gesture(hand_landmarks, handedness_str, image_width, image_height):
    """
    Classify into: 'Open Palm', 'Fist', 'Peace (V)', 'Thumbs Up', or None.
    Rules-based using finger up/down states and thumb orientation.
    """
    # finger states: index finger uses tip 8 vs pip 6, middle 12 vs 10, ring 16 vs 14, pinky 20 vs 18
    index_up = finger_is_up(hand_landmarks, 8, 6, image_height)
    middle_up = finger_is_up(hand_landmarks, 12, 10, image_height)
    ring_up = finger_is_up(hand_landmarks, 16, 14, image_height)
    pinky_up = finger_is_up(hand_landmarks, 20, 18, image_height)
    # thumb: special treatment
    thumb_ext = thumb_is_extended(hand_landmarks, handedness_str, image_width)
    thumb_up_vert = thumb_points_up(hand_landmarks, image_height)

    # Heuristics:
    # 1) Open Palm: all fingers (including thumb) are extended / up
    if index_up and middle_up and ring_up and pinky_up and thumb_ext:
        return "Open Palm"

    # 2) Fist: no fingers up, thumb not extended (all tips below pips)
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_ext:
        return "Fist"

    # 3) Peace Sign / V: index & middle up, ring & pinky down, thumb not extended
    if index_up and middle_up and not ring_up and not pinky_up:
        # thumb can be extended or not depending on style; require thumb not strongly extended sideways
        if not thumb_ext:
            return "Peace (V)"
        # sometimes people hold thumb out — still allow peace
        return "Peace (V)"

    # 4) Thumbs Up: thumb extended and pointing up, other fingers down
    if thumb_ext and thumb_up_vert and (not index_up and not middle_up and not ring_up and not pinky_up):
        return "Thumbs Up"

    return None


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Flip for natural (mirror) effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(frame_rgb)
            gesture_text = ""
            if result.multi_hand_landmarks:
                # use first detected hand
                for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    # handedness e.g. 'Left' or 'Right'
                    handedness_str = hand_handedness.classification[0].label

                    # draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0,128,255), thickness=2))

                    # bounding box (from landmarks)
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(xs) * w) - 10, int(max(xs) * w) + 10
                    y_min, y_max = int(min(ys) * h) - 10, int(max(ys) * h) + 10
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    # classify
                    gesture = classify_gesture(hand_landmarks, handedness_str, w, h)
                    if gesture:
                        gesture_text = f"{gesture}"
                        # show confidence-ish indicator: compute distance between index tip and thumb tip (just for visual)
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        dist = math.hypot((thumb_tip.x-index_tip.x)*w, (thumb_tip.y-index_tip.y)*h)
                        cv2.putText(frame, gesture_text, (x_min, y_min - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            # FPS
            ctime = time.time()
            fps = 1 / (ctime - ptime) if ptime != 0 else 0.0
            ptime = ctime
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if gesture_text:
                # big overlay center
                cv2.putText(frame, gesture_text, (int(w*0.02), int(h*0.95)),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
