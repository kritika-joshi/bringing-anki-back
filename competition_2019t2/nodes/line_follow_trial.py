#! /usr/bin/env python
from __future__ import print_function
import rospy
from geometry_msgs.msg import Twist

import roslib
roslib.load_manifest('competition_2019t2')
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



class image_converter:
    print("i am inside img converter")

    def __init__(self):
        self.image_pub = rospy.Publisher("image_pub",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw",Image,self.callback)
        self.twist = Twist()
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel_pub",Twist, queue_size=1)
 
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            print("got img")
        except CvBridgeError as e:
            rospy.loginfo(e)
        
        (h,w,channels) = cv_image.shape
        print("cv_img shape is:")
        print(cv_image.shape)

        ret,thresh2 = cv2.threshold(cv_image,190,255,cv2.THRESH_BINARY)

        cropped_thresh = thresh2[h-200:h, 0:w]
        print("cropped thresh shape is")
        print(cropped_thresh.shape)

        #converts image to gray scale
        gray_cropped_thresh = cv2.cvtColor(cropped_thresh, cv2.COLOR_BGR2GRAY)
        print("thresh2 shape is")
        print(thresh2.shape)
        print("gray_cropped_thresh is")
        print(gray_cropped_thresh.shape)

        #thresholding for one picture
        # cv_image_gray = cv2.imread(cv_image,cv2.COLOR_BGR2GRAY)
        # ret,thresh2 = cv2.threshold(cv_image_gray,190,255,cv2.THRESH_BINARY_INV)

        # cropped_thresh = gray_cropped_thresh[h-100:h, 0:w]
        # print(cropped_thresh.shape)
        cropL = gray_cropped_thresh[0:h, 0: 320]
        print("cropL shape is:")
        print(cropL.shape)
        cropR = gray_cropped_thresh[0:h, 320 : 640]
        bw1 = cropL
        bw2 = cropR



        M1 = cv2.moments(bw1)

        if int(M1['m00'] !=0 ):
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
        else:
            cX1 = w/4
            cY1 = 50
            
        M2 = cv2.moments(bw2)

        if int(M2['m00'] !=0 ):
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])
        else:
            cX2 = w/4
            cY2 = 50
        #set defaulft value if theres no m00

        print(cX1,cY1, cX2, cY2)

        cX = int((cX1 + cX2 + 320)/2)
        cY = int((cY1 + cY2)/2) +380


     
        cv2.circle(cv_image, (cX, cY + (h-200)), 5, (255, 255, 255), -1)
        cv2.putText(cv_image, "centroid", (cX - 25, (cY + (h-40)) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,  255, 255), 2)
        # BEGIN CONTROL

        
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rate = rospy.Rate(2)
        move = Twist()
        move.linear.x = 0.3
        err = cX - w/2
        print(err)
        
        
        if err == 0:
           move.angular.z = 0
           print("angular is 0")
        else:
            move.angular.z = -float(err/50)
            print("angular based on err")
            print("the err is:",err)

        pub.publish(move) 
        #END CONTROL 
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            print("published img")
        except CvBridgeError as e:
            rospy.loginfo(e)
            print("cv error")
 
def main(self):
    print("in main")
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
        print("inside image converter")
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

 
if __name__ == '__main__':
     main(sys.argv)
