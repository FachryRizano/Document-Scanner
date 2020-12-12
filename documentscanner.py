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
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 20)
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

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

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
if biggest.shape != 0:
    imgWarped = getWarp(img,biggest)
    imgArray = ([img,imgThres],
            [imgContour,imgWarped])
else:
    imgArray = ([img,imgThres],
                [img,img])
stackedImages = stackImages(0.6,imgArray)
cv2.imshow("Result",stackedImages)
cv2.waitKey(0)