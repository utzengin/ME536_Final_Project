import cv2
import numpy as np
import os
 
path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures=1000)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",35,255,empty)
cv2.createTrackbar("Threshold2","Parameters",35,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

#### Import Images
images = []
classNames= []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def boundaries(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
 
def descriptions(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList
 
def identifications(img, desList,thres=10):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal
 
desList = descriptions(images)
print(len(desList))
 
cap = cv2.VideoCapture(1)
 
while True:
 
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    #img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
 
    id = identifications(img2,desList)    

    imgContour = imgOriginal.copy()
    imgBlur = cv2.GaussianBlur(imgOriginal, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    boundaries(imgDil,imgContour)

    if id != -1:
        cv2.putText(imgContour,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

    cv2.imshow("Result", imgContour) 
    cv2.waitKey(1)