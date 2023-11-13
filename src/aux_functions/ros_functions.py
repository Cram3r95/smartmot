#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Fri July 14 00:21:19 2023
@author: Carlos Gómez Huélamo
"""

# General purpose imports

import csv
import os
import time
import pdb
import sys

# DL & Math imports

import math
import numpy as np

# ROS imports

import rospy

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from tf.transformations import quaternion_from_euler

# Custom imports

#######################################

LIFETIME = 0.25
NUM_COLOURS = 32
COLOURS = np.random.rand(NUM_COLOURS,3)
PRED_LEN = 20 # Steps
OBS_LEN = 30

def get_detection_marker(actor, global_position, is_relevant):
    """
    """

    marker = Marker()
    marker.header.frame_id = "/map"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.id = actor.object_id
    
    marker.scale.x = actor.l
    marker.scale.y = actor.w
    marker.scale.z = 1.5
    
    marker.color.a = 1.0
    
    if is_relevant:
        marker.ns = "in_monitored_area_object"
        colour = COLOURS[actor.object_id%NUM_COLOURS]
        marker.color.r = colour[0]
        marker.color.g = colour[1]
        marker.color.b = colour[2]
    else:
        marker.ns = "out_monitored_area_object"
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
    
    quaternion = quaternion_from_euler(0,0,actor.o)
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]
    
    marker.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    marker.pose.position.x = global_position.x
    marker.pose.position.y = global_position.y
    marker.pose.position.z = 0
    
    return marker

def get_trajectory_marker(obstacle_buffer):
    """
    """
    
    global_vel_lin_list = []
    global_vel_ang_list = []
    
    last_global_angle = 0
    last_global_position = []
    obstacle_id = obstacle_buffer[0].object_id

    # Observations
         
    observations_marker = Marker()
    observations_marker.header.frame_id = "/map"
    observations_marker.type = observations_marker.LINE_STRIP
    observations_marker.action = observations_marker.ADD
    observations_marker.ns = "agents_observations"
    observations_marker.id = obstacle_id
    
    observations_marker.scale.x = 0.4
    observations_marker.pose.orientation.w = 1.0
    
    colour = COLOURS[obstacle_id%NUM_COLOURS]
    observations_marker.color.r = colour[0]
    observations_marker.color.g = colour[1]
    observations_marker.color.b = colour[2]
    observations_marker.color.a = 0.5
    
    observations_marker.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    for i,obstacle in enumerate(obstacle_buffer):   
        point = Point()

        point.x = obstacle.global_position.x
        point.y = obstacle.global_position.y
        point.z = 0

        observations_marker.points.append(point)
         
        if i == len(obstacle_buffer) - 1:
            last_global_angle = obstacle.o
            last_global_position = [obstacle.global_position.x,
                                    obstacle.global_position.y,
                                    0]
        
        global_vel_lin = math.sqrt(pow(obstacle.global_velocity.x,2)+
                                   pow(obstacle.global_velocity.y,2))
        global_vel_lin_list.append(global_vel_lin)
        
        global_vel_ang_list.append(obstacle.global_angular_velocity.z)
    
    # CTRV (Constant Turn Rate Velocity) prediction
    
    global_vel_lin_array = np.array(global_vel_lin_list)
    abs_vel_lin = np.average(global_vel_lin_array)
    global_vel_ang_array = np.array(global_vel_ang_list)
    abs_vel_ang = np.average(global_vel_ang_array)
    
    predictions_marker = Marker()
    predictions_marker.header.frame_id = "/map"
    predictions_marker.type = predictions_marker.LINE_STRIP
    predictions_marker.action = predictions_marker.ADD
    predictions_marker.ns = "agents_predictions"
    predictions_marker.id = obstacle_id
    
    predictions_marker.scale.x = 0.6
    predictions_marker.pose.orientation.w = 1.0

    predictions_marker.color.r = colour[0]
    predictions_marker.color.g = colour[1]
    predictions_marker.color.b = colour[2]
    predictions_marker.color.a = 1.0
    
    predictions_marker.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    predictions_marker.points.append(point) # Last observation
    
    future_angle = last_global_angle
    
    for j in range(PRED_LEN):  
        timestep = float(j / 10) # Assuming these future steps are given at a frequency of 10 Hz
            
        point = Point()

        point.x = last_global_position[0] + abs_vel_lin * timestep * math.cos(future_angle)
        point.y = last_global_position[1] + abs_vel_lin * timestep * math.sin(future_angle)
        point.z = 0

        predictions_marker.points.append(point)
        
        future_angle = last_global_angle + abs_vel_ang * timestep
        
    return observations_marker, predictions_marker

def tracker_to_marker(tracker,color,stamp,frame_id):
    """
    Fill the obstacle features using real world metrics. Tracker presents a predicted state vector 
    (x,y,l,w,theta, ID)
    """
    tracked_obstacle = Marker()

    tracked_obstacle.header.frame_id = frame_id
    tracked_obstacle.header.stamp = stamp
    tracked_obstacle.ns = "tracked_obstacles"
    tracked_obstacle.action = tracked_obstacle.ADD
    tracked_obstacle.type = 1
        
    tracked_obstacle.id = tracker[5].astype(int)

    tracked_obstacle.pose.position.x = tracker[0] 
    tracked_obstacle.pose.position.y = tracker[1]
    tracked_obstacle.pose.position.z = 1 # TODO: Improve this
    
    tracked_obstacle.scale.x = tracker[2]
    tracked_obstacle.scale.y = tracker[3]
    tracked_obstacle.scale.z = 1.5 # TODO: Improve this
    
    quaternion = quaternion_from_euler(0,0,tracker[4])
    
    tracked_obstacle.pose.orientation.x = quaternion[0]
    tracked_obstacle.pose.orientation.y = quaternion[1]
    tracked_obstacle.pose.orientation.z = quaternion[2]
    tracked_obstacle.pose.orientation.w = quaternion[3]
        
    tracked_obstacle.color.r = color[2]
    tracked_obstacle.color.g = color[1]
    tracked_obstacle.color.b = color[0]
    tracked_obstacle.color.a = 1.0

    tracked_obstacle.lifetime = rospy.Duration(0.8) # seconds

    return tracked_obstacle

def get_detection_marker(actor):
    """
    """

    marker = Marker()
    marker.header.frame_id = "/map"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.id = actor.object_id
    
    marker.scale.x = actor.dimensions.x
    marker.scale.y = actor.dimensions.y
    marker.scale.z = actor.dimensions.z
    
    marker.color.a = 1.0

    marker.ns = "out_monitored_area_object"
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 0.5
    
    quaternion = quaternion_from_euler(0,0,actor.rotation_z)
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]
    
    marker.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    marker.pose.position.x = actor.global_position.x
    marker.pose.position.y = actor.global_position.y
    marker.pose.position.z = 1.0
    
    return marker