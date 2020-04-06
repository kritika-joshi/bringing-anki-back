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
        
        (cols,rows,channels) = cv_image.shape
        print(cv_image.shape)

        #thresholding for one picture
        ret,thresh2 = cv2.threshold(cv_image,90,255,cv2.THRESH_BINARY_INV)

        cropped_thresh = thresh2[cols-200:cols, 0:rows]
        print(cropped_thresh.shape)

        #converts image to gray scale
        gray_cropped_thresh = cv2.cvtColor(cropped_thresh, cv2.COLOR_BGR2GRAY)

        # calculate moments of binary image
        M = cv2.moments(gray_cropped_thresh)
  
        # calculate x,y coordinate of center
        #cX, cY = 0, 0
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX, cY)

        except:
            cX, cY = cols/2, rows/2
            print(cX,cY)
            

        cv2.circle(cv_image, (cX, cY + (cols-200)), 5, (255, 255, 255), -1)
        cv2.putText(cv_image, "centroid", (cX - 25, (cY + (cols-40)) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,  255, 255), 2)
        # BEGIN CONTROL

        
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rate = rospy.Rate(2)
        move = Twist()
        move.linear.x = 0.2
        err = cX - cols/2
        print(err)
        
        
        if err == 0:
           move.angular.z = 0
           print("angular is 0")
        else:
            move.angular.z = -float(err/75)
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
