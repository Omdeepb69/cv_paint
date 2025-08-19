import cv2 as cv
import mediapipe as mp
import numpy as np
import time

class Button:
    def __init__(self, x, y, width, height, label, color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color
        
    def draw(self, frame):
        cv.rectangle(frame, (self.x, self.y),
                     (self.x + self.width, self.y + self.height),
                     self.color, -1)
        cv.putText(frame, self.label,
                   (self.x + 10, self.y + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 255), 2)
        
    def is_clicked(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

def get_size(hand_landmarks, h, w):
    if not hand_landmarks:
        return 0

    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

    distance = int(np.hypot(index_x - thumb_x, index_y - thumb_y))
    
    return distance, (thumb_x, thumb_y), (index_x, index_y)

def count_fingers(hand_landmarks, h, w):
    if not hand_landmarks:
        return 0

    mp_hands = mp.solutions.hands

    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    count = 0

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if thumb_tip.x < thumb_ip.x:
        count += 1

    finger_bases = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    for tip, base in zip(finger_tips[1:], finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1

    return count

def main():
    cam = cv.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    POINTER_SIZE = 10
    draw_color = (0, 255, 0)

    canvas = None
    prev_x, prev_y = None, None

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    
    buttons = [
        Button(10, 10, 100, 50, "Red", (0, 0, 255)),
        Button(120, 10, 100, 50, "Green", (0, 255, 0)),
        Button(230, 10, 100, 50, "Blue", (255, 0, 0)),
        Button(340, 10, 150, 50, "Clear", (200, 200, 200)),
        Button(500, 10, 100, 50, "Edit", (0, 255, 255))
    ]
    
    edit_mode = False
    edit_time_start = None
    countdown_done = False

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Unable to grab frame")
            break

        frame = cv.flip(frame, 1) 
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if canvas is None:
            canvas = np.zeros_like(frame)

        results = hands.process(rgb_frame)
        
        for btn in buttons:
            btn.draw(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                fingers_up = count_fingers(hand_landmarks, h, w)

                if fingers_up == 5:
                    cv.putText(frame, "Erase Mode", (10, 90),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv.circle(frame, (x, y), POINTER_SIZE, (0, 255, 0), cv.FILLED)
                    cv.circle(canvas, (x, y), POINTER_SIZE, (0, 0, 0), cv.FILLED)

                elif fingers_up == 1:
                    clicked_button = False
                    for btn in buttons:
                        if btn.is_clicked(x, y):
                            cv.rectangle(frame, (btn.x, btn.y),
                                        (btn.x + btn.width, btn.y + btn.height),
                                        (0, 255, 255), 2)
                            if btn.label == "Clear":
                                canvas = np.zeros_like(frame)
                            elif btn.label == "Red":
                                draw_color = (0, 0, 255)
                            elif btn.label == "Green":
                                draw_color = (0, 255, 0)
                            elif btn.label == "Blue":
                                draw_color = (255, 0, 0)
                            elif btn.label == "Edit":
                                if not edit_mode:
                                    edit_mode = True
                                    edit_time_start = time.time()
                                    countdown_done = False
                                clicked_button = True

                    if not clicked_button and prev_x is not None and prev_y is not None and not edit_mode:
                        cv.line(canvas, (prev_x, prev_y), (x, y), draw_color, POINTER_SIZE)

                if edit_mode and not countdown_done:
                    elapsed = time.time() - edit_time_start
                    remaining = int(3 - elapsed)
                    distance, thumb_pos, index_pos = get_size(hand_landmarks, h, w)

                    cv.line(frame, thumb_pos, index_pos, (255, 0, 255), 2)
                    cv.putText(frame, f"Pointer Size: {distance}", (10, 170),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv.circle(frame, (50, 200), max(5, distance // 4), (0, 255, 0), -1)
                    cv.circle(frame, thumb_pos, 10, (0, 0, 255), -1)

                    if remaining > 0:
                        cv.putText(frame, f"Selecting size in {remaining}", (10, 130),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        POINTER_SIZE = distance if distance > 0 else POINTER_SIZE
                        cv.putText(frame, f"Size locked: {POINTER_SIZE}", (10, 130),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        countdown_done = True
                        edit_mode = False

                prev_x, prev_y = x, y
                if not edit_mode:
                    cv.circle(frame, (x, y), POINTER_SIZE, (0, 255, 0), cv.FILLED)

        else:
            prev_x, prev_y = None, None

        combined_frame = cv.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv.imshow("Hand Tracking", combined_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
