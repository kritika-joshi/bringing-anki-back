#! /usr/bin/env python
from __future__ import print_function

from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

import rospy
from geometry_msgs.msg import Twist

import roslib
roslib.load_manifest('competition_2019t2')
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


rospy.init_node('odom_pub')

odom_pub=rospy.Publisher ('/my_odom', Odometry)

rospy.wait_for_service ('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

odom=Odometry()
header = Header()
header.frame_id='/odom'

model = GetModelStateRequest()
model.model_name='lab_robot'

r = rospy.Rate(2)

while not rospy.is_shutdown():
    result = get_model_srv(model)
    print(result.position.x)
    print("the pos is:")
    print(result.pose)
    print("the twist is")
    print(result.twist)

    #print(result)
   # print(xcoord)

    odom.pose.pose = result.pose
    odom.twist.twist = result.twist

    header.stamp = rospy.Time.now()
    odom.header = header

    odom_pub.publish (odom)
    

    r.sleep()