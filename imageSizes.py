import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

def height_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        eye=(round(pose_results.pose_landmarks.landmark[5].x* image_width),round(pose_results.pose_landmarks.landmark[5].y* image_hight)-15)
        feet=(round(pose_results.pose_landmarks.landmark[30].x* image_width),round(pose_results.pose_landmarks.landmark[30].y* image_hight))
        cv2.circle(frame,eye,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,feet,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,eye,feet,(255,255,255),2)
        h=round(pose_results.pose_landmarks.landmark[30].y* image_hight)-round(pose_results.pose_landmarks.landmark[5].y* image_hight-15)
#         print("height=",h)
        return h

def shoulder_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        left_shoulder=(round(pose_results.pose_landmarks.landmark[11].x* image_width+8),round(pose_results.pose_landmarks.landmark[11].y* image_hight))
        right_shoulder=(round(pose_results.pose_landmarks.landmark[12].x* image_width-8),round(pose_results.pose_landmarks.landmark[12].y* image_hight))
        cv2.circle(frame,left_shoulder,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,right_shoulder,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,left_shoulder,right_shoulder,(255,255,255),2)
        shoulder_length=round(pose_results.pose_landmarks.landmark[11].x* image_width+8)-round(pose_results.pose_landmarks.landmark[12].x* image_width-8)
#         print("Shoulder_length=",shoulder_length)
        return shoulder_length

def bicepsLength_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        right_shoulder=(round(pose_results.pose_landmarks.landmark[12].x* image_width),round(pose_results.pose_landmarks.landmark[12].y* image_hight))
        right_elbow=(round(pose_results.pose_landmarks.landmark[14].x* image_width),round(pose_results.pose_landmarks.landmark[14].y* image_hight))
        cv2.circle(frame,right_shoulder,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,right_elbow,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,right_shoulder,right_elbow,(255,255,255),2)
        bicepsLength=round(pose_results.pose_landmarks.landmark[14].y* image_hight)-round(pose_results.pose_landmarks.landmark[12].y* image_hight)
#         print("bicepsLength=",bicepsLength)
        return bicepsLength

def armLength_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        right_elbow=(round(pose_results.pose_landmarks.landmark[14].x* image_width),round(pose_results.pose_landmarks.landmark[14].y* image_hight))
        right_wrist=(round(pose_results.pose_landmarks.landmark[16].x* image_width),round(pose_results.pose_landmarks.landmark[16].y* image_hight))
        cv2.circle(frame, right_elbow,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,right_elbow,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,right_wrist,right_elbow,(255,255,255),2)
        armLength=round(pose_results.pose_landmarks.landmark[16].y* image_hight)-round(pose_results.pose_landmarks.landmark[14].y* image_hight)
#         print("armLength=",armLength)
        return armLength
    
def waist_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        right_hip=(round(pose_results.pose_landmarks.landmark[24].x* image_width-23),round(pose_results.pose_landmarks.landmark[24].y* image_hight))
        left_hip=(round(pose_results.pose_landmarks.landmark[23].x* image_width+23),round(pose_results.pose_landmarks.landmark[23].y* image_hight))
        cv2.circle(frame,right_hip,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,left_hip,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,left_hip,right_hip,(255,255,255),2)
        waist=round(pose_results.pose_landmarks.landmark[23].x* image_width+23)-round(pose_results.pose_landmarks.landmark[24].x* image_width-23)
#         print("waist=",waist)
        return waist
    
def thighLength_in_image(frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)

        image_hight, image_width, _ = frame.shape

        right_hip=(round(pose_results.pose_landmarks.landmark[24].x* image_width),round(pose_results.pose_landmarks.landmark[24].y* image_hight))
        right_knee=(round(pose_results.pose_landmarks.landmark[26].x* image_width),round(pose_results.pose_landmarks.landmark[26].y* image_hight))
        cv2.circle(frame,right_hip,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,right_knee,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,right_hip,right_knee,(255,255,255),2)
        thigh_length=round(pose_results.pose_landmarks.landmark[26].y* image_hight)-round(pose_results.pose_landmarks.landmark[24].y* image_hight-20)
#         print("thigh_length=",thigh_length)
        return thigh_length
    
def legLength_in_image(thigh_length,frame):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_results=pose.process(frame_rgb)
        image_hight, image_width, _ = frame.shape
        right_knee=(round(pose_results.pose_landmarks.landmark[26].x* image_width),round(pose_results.pose_landmarks.landmark[26].y* image_hight))
        right_heel=(round(pose_results.pose_landmarks.landmark[30].x* image_width),round(pose_results.pose_landmarks.landmark[30].y* image_hight))
        cv2.circle(frame,right_knee,5,(255,255,255),cv2.FILLED)
        cv2.circle(frame,right_heel,5,(255,255,255),cv2.FILLED)
        cv2.line(frame,right_knee,right_heel,(255,255,255),2)
        legLength=round(pose_results.pose_landmarks.landmark[30].y* image_hight)-round(pose_results.pose_landmarks.landmark[26].y* image_hight-20)+thigh_length
#         print("legLength=",legLength)
        return legLength

