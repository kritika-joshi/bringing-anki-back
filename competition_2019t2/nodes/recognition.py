# has the CNN for character recgnition, takes in the pre-processed images that come from detection.py

import numpy as np
from darkflow.net.build import TFNet
import cv2


#loading the YOLO model

#when copy-pasted code from website, it was cfg/yolo-lc.cfg but that 
#doesn't exist in what we downloaded


#need to move darkflow folders into nodes I guess

options = {"model": "cfg/yolo.cfg",
           "load": "bin/yolo.weights",
           "learning_rate": 0.0000001,
           "batch": 8,
           "epoch": 25,
           "gpu": 0.6,
           "train": True,
           "annotation": "test/training/annotations/",
           "dataset": "test/training/images/"}
tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()


## -- plate detection  --

#load the weights for plate recognition
options = {"pbLoad": "yolo-plate.pb", "metaLoad": "yolo-plate.meta", "gpu": 0.9}
yoloPlate = TFNet(options)

#make the predictions with images from the processed images
predictions = yoloPlate.return_predict(frame)
firstCropImg = firstCrop(frame, predictions)
secondCropImg = secondCrop(firstCropImg)


#first crop of the license plate image
def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[i].get('topleft').get('x')
    ytop = predictions[i].get('topleft').get('y')
    xbottom = predictions[i].get('bottomright').get('x')
    ybottom = predictions[i].get('bottomright').get('y')    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img,(xtop,ytop),(xbottom,ybottom),(0,255,0),3)    
    return firstCrop


#second crop of the license plate image to reduce noise
def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        bounds = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else:
        secondCrop = img
    return secondCrop



## -- character segmentation --

#load the weights for character segmentation
options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta", "gpu":0.9}
yoloCharacter = TFNet(options)

#process the plate with bounding boxes
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edgedgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
edges = auto_canny(thresh_inv)ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
img_area = img.shape[0]*img.shape[1]for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi_area = w*h
    roi_ratio = roi_area/img_areaif((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
        if ((h>1.2*w) and (3*w>=h)):
            cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)



# def pedestrian_ahead():