"""
Functions for classifying hand gestures based on MediaPipe 21 hand landmarks
"""

import mediapipe as mp

# Helpful enumerations for finger tips and folded positions
FINGER_TIPS = [
    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    mp.solutions.hands.HandLandmark.PINKY_TIP,
]
FINGER_DIP = [
    mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
    mp.solutions.hands.HandLandmark.RING_FINGER_DIP,
    mp.solutions.hands.HandLandmark.PINKY_DIP,
]


def is_finger_open(landmarks, tip, dip):
    """
    Returns True if finger is open (tip is above dip)
    """
    return landmarks[tip].y < landmarks[dip].y


# def classify_gesture(hand_landmarks):
#     """
#     Classify gesture from landmarks
#     """
#     lm = hand_landmarks.landmark
#     tips_open = [is_finger_open(lm, tip, dip) for tip, dip in zip(FINGER_TIPS, FINGER_DIP)]
#     thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
#     thumb_ip = lm[mp.solutions.hands.HandLandmark.THUMB_IP]
#     thumb_open = thumb_tip.x < thumb_ip.x  # For right hand, use .x; for left, swap if needed

#     # Open Palm: all fingers open
#     if all(tips_open) and thumb_open:
#         return "Open Palm"
#     # Fist: all fingers folded (tips below dips), thumb folded
#     if not any(tips_open) and not thumb_open:
#         return "Fist"
#     # Peace (V): index and middle open, ring and pinky folded
#     if tips_open and tips_open[1] and not tips_open[13] and not tips_open[6]:
#         return "Peace Sign"
#     # Thumbs Up: thumb open, all others folded
#     if thumb_open and not any(tips_open):
#         return "Thumbs Up"
#     return "Unknown"

def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    tips_open = [is_finger_open(lm, tip, dip) for tip, dip in zip(FINGER_TIPS, FINGER_DIP)]
    thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp.solutions.hands.HandLandmark.THUMB_IP]
    thumb_open = thumb_tip.x < thumb_ip.x

    if all(tips_open) and thumb_open:
        return "Open Palm"
    if not any(tips_open) and not thumb_open:
        return "Fist"
    # Peace Sign: index and middle fingers open, ring and pinky folded
    if tips_open and tips_open[1] and not tips_open[2] and not tips_open[3]:
        return "Peace Sign"
    if thumb_open and not any(tips_open):
        return "Thumbs Up"
    return "Unknown"
