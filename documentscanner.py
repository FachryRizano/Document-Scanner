import cv2
import numpy as np
wImg = 480
hImg = 640


# cap = cv2.VideoCapture(0)
# cap.set(3,wImg)
# cap.set(4,hImg)
# cap.set(10,150)


def preProcessing(img):
    #make gray image
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #make blur image
    imgBlur =  cv2.GaussianBlur(imgGray,(5,5),1)
    #make canny image
    imgCanny = cv2.Canny(imgBlur,200,200)
    #Dialation function for making edges thicker and Elosion function to make thinner
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold = cv2.erode(imgDial,kernel,iterations=1)

    return imgThreshold


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),20)
            #memperkiraan sudut dari batas
            peri = cv2.arcLength(cnt,True)

            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) ==4:
                biggest = approx
                maxArea = area
    return biggest
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myNewPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print(add)

    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    return myNewPoints
def getWarp(img,biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[wImg,0],[0,hImg],[wImg,hImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(wImg,hImg))
    imgCropped = imgOutput[20:imgOutput.shape[0]-50,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(wImg,hImg))
    return imgCropped

# while True:
#     success, img = cap.read()
#     img = cv2.resize(img,(wImg,hImg))
#     imgContour = img.copy()
#
#     imgThres = preProcessing(img)
#     biggest = getContours(imgThres)
#     imgWarped = getWarp(img,biggest)
#
#     cv2.imshow("Result", imgWarped)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

img = cv2.imread("Resources/paper.jpg")

img = cv2.resize(img,(wImg,hImg))
imgContour = img.copy()

imgThres = preProcessing(img)
biggest = getContours(imgThres)
imgWarped = getWarp(img,biggest)
cv2.imshow("Result",imgWarped)
cv2.waitKey(0)