#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on tue Nov 07 11:57:13 2021
@author: Carlos Gómez Huélamo
"""

import rospy
import geometry_msgs.msg
import visualization_msgs.msg
import t4ac_msgs.msg
import numpy as np
import math

from modules.monitors.geometric_functions import euler_to_quaternion
from modules.detection_3d.estimation_3d_utils import yaw2quaternion_camera
from pyquaternion import Quaternion

def yaw2quaternion(yaw: float) -> Quaternion:
    """
    """

    return Quaternion(axis=[0,0,1], radians=yaw) # LiDAR frame

def get_quaternion(localization_pose_msg):
    """
    """
    quaternion = np.zeros((4))
    quaternion[0] = localization_pose_msg.pose.pose.orientation.x
    quaternion[1] = localization_pose_msg.pose.pose.orientation.y
    quaternion[2] = localization_pose_msg.pose.pose.orientation.z
    quaternion[3] = -localization_pose_msg.pose.pose.orientation.w

    return quaternion

def get_detection_marker(index, detection, frame_id, stamp, color=None, camera=None, lifetime=0.5):
    """
    """
    detected_3D_object_marker = visualization_msgs.msg.Marker()
    detected_3D_object_marker.header.stamp = stamp
    detected_3D_object_marker.header.frame_id = frame_id
    detected_3D_object_marker.action = visualization_msgs.msg.Marker.ADD
    detected_3D_object_marker.id = index
    detected_3D_object_marker.lifetime = rospy.Duration.from_sec(lifetime)
    detected_3D_object_marker.type = visualization_msgs.msg.Marker.CUBE
    detected_3D_object_marker.pose.position.x = detection.x
    detected_3D_object_marker.pose.position.y = detection.y
    detected_3D_object_marker.pose.position.z = detection.z
    if camera:
        q = yaw2quaternion_camera(0) # TODO: INCLUDE ORIENTATION
    else:
        q = yaw2quaternion(0) # TODO: INCLUDE ORIENTATION
    detected_3D_object_marker.pose.orientation.x = q[1] 
    detected_3D_object_marker.pose.orientation.y = q[2]
    detected_3D_object_marker.pose.orientation.z = q[3]
    detected_3D_object_marker.pose.orientation.w = q[0]
    detected_3D_object_marker.scale.x = 1
    detected_3D_object_marker.scale.y = 1
    detected_3D_object_marker.scale.z = 1

    detected_3D_object_marker.color.r = color[0]
    detected_3D_object_marker.color.g = color[1]
    detected_3D_object_marker.color.b = color[2]
    detected_3D_object_marker.color.a = color[3]

    return detected_3D_object_marker

def get_particular_monitorized_area_marker_list(particular_monitorized_area_list, closest_object, timestamp, frame_id):
    """
    """

    particular_monitorized_area_markers_list = visualization_msgs.msg.MarkerArray()

    for i, area in enumerate(particular_monitorized_area_list):
        particular_monitorized_area_marker = visualization_msgs.msg.Marker()
        particular_monitorized_area_marker.header.frame_id = frame_id
        particular_monitorized_area_marker.header.stamp = timestamp
        particular_monitorized_area_marker.ns = "particular_monitorized_areas"
        particular_monitorized_area_marker.action = particular_monitorized_area_marker.ADD
        particular_monitorized_area_marker.lifetime = rospy.Duration.from_sec(1)
        particular_monitorized_area_marker.id = i
        particular_monitorized_area_marker.type = visualization_msgs.msg.Marker.LINE_STRIP

        if closest_object == i:
            particular_monitorized_area_marker.color.r = 1.0
        else:
            particular_monitorized_area_marker.color.g = 1.0
        particular_monitorized_area_marker.color.a = 1.0 

        particular_monitorized_area_marker.scale.x = 0.3
        particular_monitorized_area_marker.pose.orientation.w = 1.0

        for p in area:
            point = geometry_msgs.msg.Point()

            point.x = p.x
            point.y = p.y
            point.z = 0.2

            particular_monitorized_area_marker.points.append(point)

        point = geometry_msgs.msg.Point()

        point.x = area[0].x
        point.y = area[0].y

        particular_monitorized_area_marker.points.append(point) # To close the polygon
        particular_monitorized_area_markers_list.markers.append(particular_monitorized_area_marker)

    return particular_monitorized_area_markers_list

def get_lane_marker(stamp,frame,scale,colour,alpha,namespace,marker_type,marker_id=0):
    """
    """

    lane_marker = visualization_msgs.msg.Marker()
    lane_marker.header.stamp = stamp
    lane_marker.header.frame_id = frame
    lane_marker.ns = namespace
    lane_marker.action = visualization_msgs.msg.Marker.ADD
    lane_marker.pose.orientation.w = 1.0
    lane_marker.id = marker_id
    lane_marker.type = marker_type
    lane_marker.lifetime = rospy.Duration.from_sec(0.4)
    lane_marker.scale.x = scale
    lane_marker.scale.y = scale
    lane_marker.color.r = colour[0]
    lane_marker.color.g = colour[1]
    lane_marker.color.b = colour[2]
    lane_marker.color.a = alpha

    return lane_marker

def get_traffic_lights_marker(traffic_lights, timestamp, rgb = [0,0,0], name = "", scale=[2,1,1], lifetime=0):
    """
    """

    traffic_lights_markers_list = visualization_msgs.msg.MarkerArray()

    for i, traffic_light in enumerate(traffic_lights):
        traffic_light_marker = visualization_msgs.msg.Marker()
        traffic_light_marker.header.frame_id = "map" # Since we are using global coordinates 
                                                     # for the traffic signals (either STOP or TL)
        traffic_light_marker.header.stamp = timestamp
        traffic_light_marker.ns = "nodes_marker" + name
        traffic_light_marker.action = traffic_light_marker.ADD
        traffic_light_marker.lifetime = rospy.Duration(lifetime)
        traffic_light_marker.id = i
        traffic_light_marker.type = visualization_msgs.msg.Marker.ARROW

        # Pose (Position and orientation)

        traffic_light_marker.pose.position.x = traffic_light.global_location.pose.position.x
        traffic_light_marker.pose.position.y = traffic_light.global_location.pose.position.y # -y since RVIZ representation is right-handed rule
        traffic_light_marker.pose.position.z = traffic_light.global_location.pose.position.z

        traffic_light_marker.pose.orientation.x = traffic_light.global_location.pose.orientation.x
        traffic_light_marker.pose.orientation.y = traffic_light.global_location.pose.orientation.y
        traffic_light_marker.pose.orientation.z = traffic_light.global_location.pose.orientation.z
        traffic_light_marker.pose.orientation.w = traffic_light.global_location.pose.orientation.w # TODO: Opposite to represent in RVIZ?
        
        # Scale

        traffic_light_marker.scale.x = scale[0] # Shaft diameter
        traffic_light_marker.scale.y = scale[1] # Head diameter
        traffic_light_marker.scale.z = scale[2] # Arrow length

        # Colour

        traffic_light_marker.color.r = rgb[0]
        traffic_light_marker.color.g = rgb[1]
        traffic_light_marker.color.b = rgb[2]
        traffic_light_marker.color.a = 1.0

        traffic_lights_markers_list.markers.append(traffic_light_marker)

    return traffic_lights_markers_list