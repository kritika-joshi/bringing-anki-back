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
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

'''this part defines your state? i think '''

#rospy.wait_for_service('/gazebo/get_model_state')
get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
model = GetModelStateRequest()
model.model_name='lab_robot'
#print("started")

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
        # self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        #self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1, latch=False)
        self.name = 'lab_robot'
        self.get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)
    
        self.get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])
        #print(" in init")


    def get_state(self, msg):
        #print("in get state")

        '''
            @bried Gets state from location of robot
            @param data: location of state from ROS
            @retval (state, done)

        '''
        # time = rospy.Time.now()
        # rospy.wait_for_service('/gazebo/get_model_state')
        try:
            modelstate = self.get_pose(model)
        except:
            print("cannot get state :( sry")
        xcoord = modelstate.pose.position.x
        ycoord = modelstate.pose.position.y
        current_rpy = euler_from_quaternion((modelstate.pose.orientation.x, modelstate.pose.orientation.y, modelstate.pose.orientation.z, modelstate.pose.orientation.w))
        angle = current_rpy[2]
        #print(xcoord, ycoord)

        img = cv2.imread("/home/fizzer/Desktop/map_mask1.jpg")
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

        #CODE TO TRANSFORM MATRIX:
        x = np.matrix(([xcoord, ycoord]))
        x1 = -(x)
        R = -np.matrix(([0,-1],[-1,0]))
        #flip = np.matrix(([0,1],[1,0]))
        x2 = (x1*R)
        x3 = (x2 +2.9)*(863/5.8)
        #print(x3)

        img_x = (x3.A1[0])
        img_y = (x3.A1[1]) 
   
        #adding a transformation to get to the nose
        img_x = int(img_x +4.2*math.sin(angle))
        img_y = int(img_y + 4.2*math.cos(angle))


        if(angle > 0):
            xorientation = "right"
        else:
            xorientation = "left"

        if(abs(angle) > math.pi/2):
            yorientation = "down"
        if(abs(angle) < math.pi/2):
            yorientation = "up"
 
        img_array = img[img_y, img_x]
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
        elif(max_index == 0):
                color = 'red'
        #print(color)

        section = 'no section'
        if(img_y in ymiddle):
            section = '3'
        elif(img_x in xright and img_y in ybot):
            section = '1'
        elif(img_x in xleft and img_y in ybot):
            section = '2'
        elif (img_x in xleft and img_y in ytop):
            section = '4'
        elif(img_x in xright and img_y in ytop):
            section = '5'
        #print("the section is:")
        #print(section)
       
        if(color == 'road'):
            state_array = [0,0,1,0,0]
            self.timeout = 0
        elif(color == 'red'):
            state_array = [1,0, 0, 0, 1]
            self.timeout += 1
        else:
            # section = 'no section'
            # if(ycoord in ymiddle):
            #     section = '3'
            # elif(xcoord in xright and ycoord in ybot):
            #     section = '1'
            # elif(xcoord in xleft and ycoord in ybot):
            #     section = '2'
            # elif (xcoord in xleft and ycoord in ytop):
            #     section = '4'
            # else:
            #     section = '5'
            
            if (section == '1' or section == '3'):
                #print("in if loop1")

                if(color == 'blue'):
                    #print("in if loop2")
                    if(xorientation == 'left' or yorientation == 'down'):
                        #print("in if loop3")
                        state_array = [0, 1, 0, 0, 0]
                    else:
                        state_array = [0, 0, 0, 1, 0]
                else:
                    if(xorientation == 'left' or yorientation == 'down'):
                        state_array = [0, 0, 0, 1, 0]
                    else:
                        state_array = [0, 1, 0, 0, 0]
            if(section =='2'):
                if (color == 'blue'):
                    if(xorientation =='left' or yorientation =='up'):
                        state_array = [0, 1, 0, 0, 0]
                    else:
                        state_array = [0,0,0,1,0]
                else:
                    if(xorientation =='left' or yorientation =='up'):
                        state_array = [0,0,0,1,0]
                    else:
                        state_array = [0,1,0,0,0]
            if (section =='4' or section =='5'):
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
            #print(state_array)

        #print(self.timeout)
        if self.timeout > 500 :
            done = True
        else:
            done = False 
        # print("get state")
        # print(done)
        return state_array, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print("inside step")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()
        #print(action)
        if action == 0:  # FORWARD
            vel_cmd.linear.x = 1.0
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.45
            vel_cmd.angular.z = 0.8
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.45
            vel_cmd.angular.z = -0.8

        self.vel_pub.publish(vel_cmd)

        msg = None

        # while msg is None:
        #     print("inside while msg is none loop 1")
        #     try:
        #        # msg = rospy.wait_for_message('/pi_camera/image_raw', Image,
        #        #                              timeout=5)
        #         #msg = roslib.wait_for_message('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        #         msg = rospy.wait_for_service ('/gazebo/get_model_state')
        #     except:
        #         pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.get_state(msg)
        # print("aha! im out of get_state")
        #print(state)

        # Set the rewards for your action
        if action == 0:  # FORWARD
            reward = 50
        elif action == 1:  # LEFT
            reward = 35
        elif action ==2:
            reward = 35  # RIGHT
        # else:
        #     reward = -200

        if(state == [1,0,0,0,1]):
            reward = -200
        # elif(state ==[0,0,1,0,0]):
        #     reward = 50
        # else:
        #     reward = -7

        #
        #print(reward)
        # print("step")
        # print(done)
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
        # while msg is None:
        #     print("in while msg is none loop")
        # msg = None
        # while msg is None:
        #     print("inside while msg is none loop reset")
        #     try:
        #        # msg = rospy.wait_for_message('/pi_camera/image_raw', Image,
        #        #                              timeout=5)
        #         #msg = roslib.wait_for_message('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        #         msg = rospy.wait_for_service('/gazebo/get_model_state')
        #         print(msg)
        #     except:
        #         pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.get_state(msg)
        # print("reset")
        # print(done)

        return state