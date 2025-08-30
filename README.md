# Real-Time Static Hand Gesture Recognition
**Author:** <Your Full Name>

## Assignment
This project implements real-time recognition of 4 static hand gestures (Open Palm, Fist, Peace Sign, Thumbs Up) from webcam input. (Assignment details provided separately.) :contentReference[oaicite:1]{index=1}

---

## Technology Justification
- **MediaPipe Hands**: provides 21 3D hand landmarks per hand, is optimized for real-time CPU inference, and is simple to integrate. It avoids heavy training and is robust to many hand poses.
- **OpenCV**: for webcam capture, drawing landmarks, and displaying the output window.
- **Rules-based logic**: using landmark geometry keeps the classifier interpretable, quick to run (no model training required), and deterministic — perfect for this assignment.

---

## Gesture Logic Explanation
Gesture recognition uses landmark geometry (21 points):
- **Open Palm**: index, middle, ring, pinky tips are above respective pip joints (y_tip < y_pip) and thumb is extended sideways — all five fingers considered up.
- **Fist**: none of the fingers' tips are above their pip joints; thumb not extended.
- **Peace (V)**: index & middle fingers up, ring & pinky down. Thumb state optional.
- **Thumbs Up**: thumb is extended and thumb tip's y is above wrist's y (points upward). The other four fingers are folded (tips below pip).

Note: image coordinates origin is top-left; smaller y means higher in the image.

---

## Environment Setup & Execution Instructions

1. **Clone repo**:
   ```bash
   git clone <your-repo-url>
   cd your-repo-folder
