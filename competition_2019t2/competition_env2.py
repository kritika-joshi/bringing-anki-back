#!/usr/bin/env python

from geometry_msgs.msg import Twist, PoseStamped, Pose
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates
import math
import numpy as np
import random
import rospy
import roslib
roslib.load_manifest('competition_2019t2')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import re
import os
import tensorflow as tf
import cv2
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import glob
import array
from numpy import argmax
from numpy import array
from array import array 
from PIL import Image
import gym
import roslaunch
import time
from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from time import sleep
from gym.utils import seeding

'''this part defines your state? i think '''

class Gazebo_Competion_2019t2_env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        try:
            LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo/gym_gazebo/envs/ros_ws/src/competition_2019t2/launch/my_launch.launch'
            print("launch file found")
        except:
            print("cannot location launch file")
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        #self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1, latch=False)
        self.name = 'lab_robot'

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])

    def pose_feedback_callback(self, msg):
        
        if(msg is not None):
            if(self.name in msg.name):
                index = msg.name.index(self.name)
                self.pose = msg.pose[index]
                if self.pose is not None:
                    current_rpy = euler_from_quaternion((self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w))
                    
                    global angle = current_rpy[2]
                    global xcoord = self.pose.position.x
                    global ycoord = self.pose.position.y
                    print("stuck in callback")
        print("gonna get out of callback")
        return xcoord, ycoord,angle 

    def get_state(self, msg, xcoord, ycoord, angle):
        print("in get state")
        '''
            @bried Gets state from location of robot
            @param data: location of state from ROS
            @retval (state, done)
        '''
        img = cv2.imread("/home/fizzer/Desktop/map_mask.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[0:865, 0:865]
        boundary12 = int(865/2)
        boundary34 = 380
        boundary23 = 520
        boundarymax = 865
        xright = list(range(boundary12, boundarymax+1))
        xleft = list(range(0,boundary12))
        ytop = list(range(0, boundary34))
        ymiddle = list(range(boundary34, boundary23))
        ybot = list(range(boundary23, boundarymax))

        #now we define our state
        state_array = [0, 0, 0, 0, 0]
        done = False

        if(msg is not None):
            if(self.name in msg.name):
                index = msg.name.index(self.name)
                self.pose = msg.pose[index]
                if self.pose is not None:
                    #current_rpy = euler_from_quaternion((self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w))
                    
                    #angle = current_rpy[2]
                    #xcoord = self.pose.position.x
                    #ycoord = self.pose.position.y

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
                    # print(xcoord)
                    # print(ycoord)
                    # print(img_x)
                    # print(img_y)
                    # print(angle)

                    if(angle > 0):
                        xorientation = "right"
                    else:
                        xorientation = "left"

                    if(abs(angle) > math.pi/2):
                        yorientation = "down"
                    if(abs(angle) < math.pi/2):
                        yorientation = "up"
                    # print(xorientation)
                    # print(yorientation)
                    img_array = img[img_y,img_x]
                    img_list = img_array.tolist()
                    max_value = max(img_list)
                    max_index = img_list.index(max_value)

                    #Get color
                    if(max_value == 0):
                        color = 'road'
                    elif(max_index ==2):
                        color = 'blue'
                    elif(max_index ==1):
                            color = 'Green'
                    elif(max_index ==1)
                            color = 'white'
                    print(color)

                    if(color == 'road'):
                        state_array = [0,0,1,0,0]
                    elif(color == 'white'):
                        state_array = [1,0, 0, 0, 0]
                    else:
                        section = 'no section'
                        if(ycoord in ymiddle):
                            section = '3'
                        elif(xcoord in xright and ycoord in ybot):
                            section = '1'
                        elif(xcoord in xleft and ycoord in ybot):
                            section = '2'
                        elif (xcoord in xleft and ycoord in ytop):
                            section = '4'
                        else:
                            section = 5
                      
                        if (section == '1' or section =='3'):
                            if(color == 'blue'):
                                if(xorientation == 'left' or yorientation == 'down'):
                                    state_array = [0,1, 0, 0,0]
                                else:
                                    state_array = [0,0, 0, 1,0]
                            else:
                                if(xorientation == 'left' or yorientation == 'down'):
                                    state_array = [0,0, 0, 1,0]
                                else:
                                    state_array = [0,1, 0, 0,0]
                        if(section =='2'):
                            if (color == 'blue'):
                                if(xorientation =='left' or yorientation =='up'):
                                    state_array = [0,1,0,0,0]
                                else:
                                    state_array = [0,0,0,1,0]
                            else:
                                if(xorientation =='left' or yorientation =='up'):
                                    state_array = [0,0,0,1,0]
                                else:
                                    state_array = [0,1,0,0,0]
                        if (section =='4' or section ==5):
                            if (color == 'blue'):
                                if(xorientation =='right' or yorientation =='up'):
                                    state_array = [0,1,0,0,0]
                                else:
                                    state_array = [0,0,0,1,0]
                            else:
                                if(xorientation =='left' or yorientation =='up'):
                                    state_array = [0,0,0,1,0]
                                else:
                                    state_array = [0,1,0,0,0]

        print(state_array)
        if self.timeout > 30 :
            done = True

        return state_array, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        print("inside step")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()
        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        msg = None
        while msg is None:
            try:
               # msg = rospy.wait_for_message('/pi_camera/image_raw', Image,
               #                              timeout=5)
                msg = roslib.wait_for_message('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.get_state(msg, xcoord, ycoord)
        print("aha! im out of get_state")
        print(state)

        # Set the rewards for your action
        if action == 0:  # FORWARD
            reward = 10
        elif action == 1:  # LEFT
            reward = 3
        else:
            reward = 3  # RIGHT

        if(state == [1,0,0,0,0]):
            reward = -5
        else:
            reward = -5

        print(reward)

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        msg = None
        while msg is None:
            try:
               # msg = rospy.wait_for_message('/pi_camera/image_raw', Image,
               #                              timeout=5)
                msg = roslib.wait_for_message('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
 
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.get_state(msg, xcoord, ycoord, angle)

        return state