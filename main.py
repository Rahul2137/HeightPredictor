import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import cvzone
# from cvzone.FaceMeshModule import FaceMeshDetector
import time
from imageSizes import height_in_image, shoulder_in_image, waist_in_image, bicepsLength_in_image, thighLength_in_image, legLength_in_image, armLength_in_image
# plt.rcParams["figure.figsize"] = (10,6)
# fig = plt.figure()
# ax = fig.add_subplot()
# fig.subplots_adjust(top=0.85)
# plt.axis([0, 20, 0, 190])
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)



def find_real_measurements(img,h,shoulder_length,thigh_length,legLength,waist,armLength,bicepsLength):
    #using distance between pupils
    frame_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pose_results=pose.process(frame_rgb)
    image_hight, image_width, _ = img.shape
    pointLeft=(round(pose_results.pose_landmarks.landmark[2].x* image_width),round(pose_results.pose_landmarks.landmark[2].y* image_hight))
    pointRight=(round(pose_results.pose_landmarks.landmark[5].x* image_width),round(pose_results.pose_landmarks.landmark[5].y* image_hight))
    cv2.line(img,pointLeft,pointRight,(0,255,255),2)
    cv2.circle(img,pointLeft,4,(0,0,255),cv2.FILLED)
    cv2.circle(img,pointRight,4,(0,0,255),cv2.FILLED)
    w=(pose_results.pose_landmarks.landmark[2].x* image_width)-(pose_results.pose_landmarks.landmark[5].x* image_width)
    eyes_width.append(w)
     
#     using face-width
#     face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces=face_detector.detectMultiScale(gray_image,1.05,5)
#     face_width=0
#     for(x,y,hf,w) in faces:
#         cv2.rectangle(img,(x+20,y+20),(x+w-20, y+hf-20),(255,255,255),3)
#         face_width=w
    if(w<12.5): W=5.69
    elif(w>12.5 and w<13.5): W=5.78
    elif(w>13.5 and w<14.5): W=6.28
    elif(w>14.5 and w<15.5): W=6.5
    elif(w>15.5):W=6.7
#     else: W=6.2
#     print(face_width)
    real_height=h*W/w
    real_shoulder_length=shoulder_length*W/w
    real_thigh_length=thigh_length*W/w
    real_leg_length=legLength*W/w
    real_waist=2*waist*W/(w*2.54)#in inches
    real_armLength=armLength*W/w
    real_bicepsLength=bicepsLength*W/w
    print(h,w,real_height,W)
    cvzone.putTextRect(img,f'height:{round(real_height,2)}cm',(10,50),
                    scale=1.9)
#     cvzone.putTextRect(img,f'shoulderflength:{round(real_shoulder_length,2)}cm',(10,100),
#                     scale=1.5)
    cvzone.putTextRect(img,f'waist:{round(real_waist,2)}inches',(10,200),
                    scale=1.9)
    
#     cvzone.putTextRect(img,f'thighlength:{round(real_thigh_length,2)}cm',(10,300),
#                     scale=1.9)
#     cvzone.putTextRect(img,f'leglength:{round(real_leg_length,2)}cm',(10,400),
#                     scale=1.9)
#     cvzone.putTextRect(img,f'armlength:{round(real_armLength,2)}cm',(10,230),
#                     scale=1.9)
#     cvzone.putTextRect(img,f'bicepsLength:{round(real_bicepsLength,2)}cm',(10,260),
#                     scale=1.9)
    return real_height,real_shoulder_length,real_thigh_length,real_leg_length,real_waist,real_armLength,real_bicepsLength


# ,static_image_mode=True,enable_segmentation=True,model_complexity=2

cap=cv2.VideoCapture(0)
time.sleep(5)
eyes_width=[]
heightImgArr=[]
real_height_array=[]
real_shoulder_array=[]
count=50
sumHeight=0
sumShoulder=0
sumWaist=0
sumThigh=0
sumLeg=0
sumArm=0
sumBiceps=0
while count>=0:

    _,frame=cap.read()

    try:        
        h=height_in_image(frame)
        heightImgArr.append(h)
        shoulder_length=shoulder_in_image(frame)
        thigh_length=thighLength_in_image(frame)
        legLength=legLength_in_image(thigh_length,frame)
        waist=waist_in_image(frame)
#         wingspan=wingspan_in_image(frame)
        armLength=armLength_in_image(frame)
        bicepsLength=bicepsLength_in_image(frame)
        real_height,real_shoulder_length,real_thigh_length,real_leg_length,real_waist,real_armLength,real_bicepsLength=find_real_measurements(frame,h,shoulder_length,thigh_length,legLength,waist,armLength,bicepsLength)     
        real_height_array.append(real_height)
        real_shoulder_array.append(real_shoulder_length)
        cv2.imshow('output',frame)
        time.sleep(0.4)
        sumHeight+=real_height
        sumShoulder+=real_shoulder_length
        sumWaist+=real_waist
        sumThigh+=real_thigh_length
        sumLeg+=real_leg_length
        sumArm+=real_armLength
        sumBiceps+=real_bicepsLength
        count-=1
    except:
        continue
    if cv2.waitKey(1)==ord('q'):
        break

avgHeight=sumHeight/50
avgShoulder=sumShoulder/50
avgWaist=sumWaist/50
avgThigh=sumThigh/50
avgLeg=sumLeg/50
avgArm=sumArm/50
avgBiceps=sumBiceps/50
avgeyewidth=sum(eyes_width)/len(eyes_width)
avgheightimg=sum(heightImgArr)/len(heightImgArr)
print("avg==",avgheightimg,avgeyewidth)
print("height by aniket=",avgheightimg*6.5/avgeyewidth)
print(f'avgheight={round(avgHeight,2)},avgshoulder={round(avgShoulder,2)},avgWaist={round(avgWaist,2)},avgThigh={round(avgThigh,2)},avgLeg={round(avgLeg,2)},avgArm={round(avgArm,2)},avgBiceps={round(avgBiceps,2)}')
# print("\neyes_width=",eyes_width)
# print("\navgeyes_width=",avgeyewidth)
# print("\nreal_height_array=",real_height_array)
# X=merge(eyes_width,heightImgArr)
plt.scatter(eyes_width,real_height_array, label= "", color= "green", 
            marker= "o", s=50)
plt.scatter(eyes_width,real_shoulder_array, label= "", color= "red", 
            marker= "o", s=50)
c, d = np.polyfit(eyes_width,real_shoulder_array, 1)
a, b = np.polyfit(eyes_width,real_height_array, 1)
print("real height by equation==>>",int(avgeyewidth)*a+b)
arr=np.array(eyes_width)
plt.plot(arr, a*(arr)+b)
plt.plot(arr, c*(arr)+d)
plt.text(arr[0],real_shoulder_array[0]+30, 'height=>y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=10)
plt.text(arr[0],real_shoulder_array[0]+25, 'shoulder=>y = ' + '{:.2f}'.format(d) + ' + {:.2f}'.format(c) + 'x', size=10)
plt.xlabel("eyes")
plt.ylabel("height")
plt.show()
cv2.destroyAllWindows()
cap.release()