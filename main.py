import cv2
import numpy as np
import dlib
from typing import Tuple





def midpoint(p1, p2) -> Tuple[int, int]:
    return (p1.x + p2.x)//2 , (p1.y + p2.y)//2

def line_length(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # first coordinate is the top-left and the second one is bottom-right.
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 =  face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1),(0,0,255),2)
        landmarks = predictor(gray, face)
        # print(face)
        # print(landmarks)
        # right_eye_x = landmarks.part(36).x
        # right_eye_y = landmarks.part(36).y
        # cv2.circle(frame,(right_eye_x,right_eye_y),3,(0,255,0))
        eye_left_point = (landmarks.part(36).x, landmarks.part(36).y)
        eye_right_point = (landmarks.part(39).x, landmarks.part(39).y)
        eye_top_point = midpoint(landmarks.part(37),landmarks.part(38))
        eye_bottom_point = midpoint(landmarks.part(40),landmarks.part(41))
        hor_line = cv2.line(frame,eye_left_point,eye_right_point,(0,0,255),2)
        ver_line = cv2.line(frame, eye_top_point
                            ,eye_bottom_point,(0,0,255),2)

        ver_line_length = line_length(eye_top_point, eye_bottom_point)
        hor_line_length = line_length(eye_left_point, eye_right_point)

        print(f"{hor_line_length/ver_line_length:.3f}")
        ratio = hor_line_length/ ver_line_length
        if ratio > 5.3:
            cv2.putText(frame,"BLINK",(50,150),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255,0,0))

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # pressing ESC will exit the program and collapse all the windows.
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()