#!/usr/bin/env python
import math
import numpy as np
import random
import rospy
import roslib
roslib.load_manifest('competition_2019t2')

from geometry_msgs.msg import Twist, PoseStamped, Pose
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates

class GetPos:

    def __init__(self):
        self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1, latch=False)
        self.pose = None
        self.heading_deadband = 0.2
        self.position_deadband = 0.015
        self.name = 'lab_robot'
        self.max_angular_vel = 2
        self.max_linear_vel = 0.2
        self.at_rest = False
        self.last_reached_dest_time = rospy.Time.now()


    def pose_feedback_callback(self, msg):
        if(msg is not None):
            if(self.name in msg.name):
                index = msg.name.index(self.name)
                self.pose = msg.pose[index]
                if self.pose is not None:
                    current_rpy = euler_from_quaternion((self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w))
                    
                    angle = current_rpy
                    xcoord = self.pose.position.x
                    ycoord = self.pose.position.y

                    #do transformation matrix to map it to the image:
                    #CODE TO TRANSFORM MATRIX:
                    x = np.matrix(([xcoord, ycoord]))
                    x1 = -(x)
                    R = -np.matrix(([0,-1],[-1,0]))
                    flip = np.matrix(([0,1],[1,0]))
                    x2 = (x1*R)
                    x3 = (x2 +2.9)*(863/5.8)
                    #print(x3)
                    img_x = int(x3.A1[0])
                    img_y = int(x3.A1[1]) 
                    print(xcoord)
                    print(ycoord)
                    print(img_x)
                    print(img_y)




                    # while angle >= math.pi:
                    #     angle = angle - 2*math.pi
                    # while angle <= -math.pi:
                    #     angle = angle + 2*math.pi
                    #dist = math.hypot(dy,dx)

                    # print(angle[2])
                    # print(self.pose.position.x)

                    # Rotate to the direct goal heading
                    # if abs(angle) > self.heading_deadband and dist > self.position_deadband and not self.at_rest:
                    #     # print("Aligning with heading")
                    #     if angle > self.heading_deadband:
                    #         cmd_vel.angular.z = -self.max_angular_vel
                    #     elif angle < -self.heading_deadband:
                    #         cmd_vel.angular.z = self.max_angular_vel
                    # Drive forwards to goal position
                    # elif dist > self.position_deadband:
                    #     # print("Moving to destination position",dist)
                    #     cmd_vel.linear.x = self.max_linear_vel * random.uniform(0.1, 1)
                    # Rotate to align with goal orientation
                    # elif dist < self.position_deadband and abs(current_rpy[2] - dest_rpy[2]) < self.orientation_deadband and not self.at_rest:
                    #     print("Aligning with desintation orientation")
                    #     if current_rpy[2] - dest_rpy[2] > 0:
                    #         cmd_vel.angular.z = -self.max_angular_vel
                    #     else:
                    #         cmd_vel.angular.z = self.max_angular_vel
                    # else:
                    #     # print("Reached goal")
                    #     if not self.at_rest:
                    #         self.last_reached_dest_time = rospy.Time.now()
                    #         self.at_rest = True
                    #     elif rospy.Time.now() - self.last_reached_dest_time > rospy.Duration(3) and self.at_rest:
                    #         self.at_rest = False
                    #         self.pose_goal_buffer[0], self.pose_goal_buffer[-1] = self.pose_goal_buffer[-1], self.pose_goal_buffer[0]
                    #         self.pose_goal = self.pose_goal_buffer[0]

                    # self.vel_pub.publish(cmd_vel)
                    #                     cmd_vel = Twist()
                    # cmd_vel.angular.z = 0
                    # cmd_vel.linear.x = 0
    # print(x)
    #print(self.pose.position.x)



if __name__ == '__main__':
    rospy.init_node('get_pos')
    cw = GetPos()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass