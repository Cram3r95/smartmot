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
import git
import time

# DL & Math imports

import math
import numpy as np

# ROS imports

import rospy

from visualization_msgs.msg import MarkerArray
from ad_perdevkit.msg import GT_3D_Object_list
from t4ac_msgs.msg import MonitorizedLanes
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion

# Custom imports

repo = git.Repo(__file__, search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from src.aux_functions import monitors_functions
from src.aux_functions import sort_functions
from src.aux_functions import ros_functions

#######################################

HEADER_SYNCHRO = 20
SLOP = 1.0
LIFETIME = 1.0
NUM_COLOURS = 32
COLOURS = np.random.rand(NUM_COLOURS,3)
PRED_LEN = 20 # Steps
OBS_LEN = 30

class SmartMOT():
    def __init__(self):
        """
        """

        self.DEBUG = False
        self.ego_vehicle_location = None
        self.trackers = {}
        self.monitored_lanes = MonitorizedLanes()
        self.COMPUTE_MARKERS = True
        self.USE_HDMAP = False
        
        # Multi-Object Tracking
        
        self.mot_tracker = sort_functions.Sort(max_age=3,min_hits=2, iou_threshold=0.1)
        self.list_of_ids = {}
        self.state = {}
        self.timestamp = 0
        self.range = 150
        self.max_agents = 11 # including ego-vehicle (only for Decision-Making module)
        
        # ROS communications
        
        self.frame_id = None
        
        ## Publishers
        
        map_monitor_filtered_detections_markers = "/t4ac/perception/detection/map_monitor_filtered_detections"
        self.pub_map_monitor_filtered_detections_markers = rospy.Publisher(map_monitor_filtered_detections_markers, MarkerArray, queue_size=10)
    
        self.pub_mot_markers = rospy.Publisher("/t4ac/perception/mot_bounding_bboxes", MarkerArray, queue_size=10)
        
        ## Subscribers
        
        localization_pose_topic = "/t4ac/localization/pose"
        self.sub_location_ego = rospy.Subscriber(localization_pose_topic, Odometry, self.localization_callback)

        # detections_topic = "/ad_devkit/generate_perception_groundtruth_node/perception_groundtruth"
        # self.sub_detections = rospy.Subscriber(detections_topic, GT_3D_Object_list, self.detections_callback)
        detections_topic = "/t4ac/perception/3D_lidar_obstacles_markers"
        self.sub_detections = rospy.Subscriber(detections_topic, MarkerArray, self.detections_callback)
        
        monitored_lanes_topic = "/t4ac/perception/monitors/monitorized_lanes"
        self.sub_monitored_lanes = rospy.Subscriber(monitored_lanes_topic, MonitorizedLanes, self.monitored_lanes_callback)
        
    def localization_callback(self, location_msg):
        """
        """
    
        self.ego_vehicle_location = location_msg
    
    def monitored_lanes_callback(self, monitored_lanes_msg):
        """
        """
        
        self.monitored_lanes = monitored_lanes_msg
    
    # TODO: Include here SmartMOT or even from the detection stage
      
    def detections_callback(self, detections_rosmsg):
        """
        """
        
        start = time.time()
        start__ = time.time()
        if self.ego_vehicle_location:
            orientation_q = self.ego_vehicle_location.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (ego_roll, ego_pitch, ego_yaw) = euler_from_quaternion(orientation_list)
                        
            map_monitor_filtered_objects = MarkerArray()
            
            relevant_detections = []
            
            # If AD-PerDevKit (at this moment in the CARLA Leaderboard)
            
            # for i,detection in enumerate(detections_rosmsg.gt_3d_object_list):
            #     if i > 0: # We assume the first object, if using GT, it is the ego-vehicle. Avoid this.
            #         detection.rotation_z = ego_yaw - detection.rotation_z # To obtain the angle in global frame
                            
            #                 if self.USE_HDMAP:
            #                     is_relevant = False
            #                     for lane in monitored_lanes_rosmsg.lanes:
            #                         if len(lane.left.way)> 1:
            #                             is_relevant, in_road, particular_monitorized_area, _, _ = monitors_functions.inside_lane(detection.global_position, lane, detection.type)
                                    
            #                             if is_relevant:
            #                                 # if detection.object_id not in self.trackers.keys():
            #                                 #     self.trackers[detection.object_id] = [detection]
            #                                 # else:
            #                                 #     self.trackers[detection.object_id].append(detection)
                                            
            #                                 # Local coordinates
            #                                 # TODO: Fill the velocity of the object in the 7th position of the list
                                            
            #                                 relevant_detection = [detection.position.x, detection.position.y, detection.position.z, # x,y,z
            #                                                       detection.dimensions.x, detection.dimensions.y, detection.dimensions.z, # l,w,h
            #                                                       detection.rotation_z, 0, detection.type] # theta, vel, type
                                            
            #                                 relevant_detections.append(relevant_detection)
                                            
            #                                 break
            #                 else:
            #                     relevant_detection = [detection.position.x, detection.position.y, detection.position.z, # x,y,z
            #                                           detection.dimensions.x, detection.dimensions.y, detection.dimensions.z, # l,w,h
            #                                           detection.rotation_z, 0, detection.type] # theta, vel, type
                                            
            #                     relevant_detections.append(relevant_detection)
            
            # If Object detection in AIVATAR project
            
            for i, detection in enumerate(detections_rosmsg.markers):
                # detection.rotation_z = ego_yaw - detection.rotation_z # To obtain the angle in global frame
                if not self.frame_id: self.frame_id = detection.header.frame_id
                
                quat_xyzw = detection.pose.orientation
                quaternion = np.array((quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w))
                detection_yaw = euler_from_quaternion(quaternion)[2]
                
                if self.USE_HDMAP:
                    is_relevant = False
                    for lane in monitored_lanes_rosmsg.lanes:
                        if len(lane.left.way)> 1:
                            is_relevant, in_road, particular_monitorized_area, _, _ = monitors_functions.inside_lane(detection.global_position, lane, detection.type)
                        
                            if is_relevant:
                                # if detection.object_id not in self.trackers.keys():
                                #     self.trackers[detection.object_id] = [detection]
                                # else:
                                #     self.trackers[detection.object_id].append(detection)
                                
                                # Local coordinates
                                # TODO: Fill the velocity of the object in the 7th position of the list
                                
                                relevant_detection = [detection.pose.position.x, detection.pose.position.y, detection.pose.position.z, # x,y,z
                                          detection.scale.x, detection.scale.y, detection.scale.z, # l,w,h
                                          detection_yaw, 0, "generic"] # theta, vel, type
                                
                                relevant_detections.append(relevant_detection)
                                
                                break
                else:
                    relevant_detection = [detection.pose.position.x, detection.pose.position.y, detection.pose.position.z, # x,y,z
                                          detection.scale.x, detection.scale.y, detection.scale.z, # l,w,h
                                          detection_yaw, 0, "generic"] # theta, vel, type
                                
                    relevant_detections.append(relevant_detection)
                
        #################################################################
        ############## MULTI-OBJECT TRACKING PIPELINE ###################
        #################################################################
        
        if self.DEBUG: print(f">>> 4. Multi-Object Tracking processing --------------- >>>")
        mott1 = time.time()

        merged_objects, types = sort_functions.merged_bboxes_to_xylwthetascore_types(relevant_detections)
        if self.DEBUG: print("Merged objects: ", merged_objects)

        trackers, types, vels = self.mot_tracker.update(merged_objects, types, self.DEBUG)

        if self.DEBUG: print("\033[1;35m"+"Final Trackers: "+'\033[0;m', trackers)
        # stamp = detections_rosmsg.header.stamp
        stamp = self.ego_vehicle_location.header.stamp
        tracker_marker_list = MarkerArray()

        for tracker in trackers:
            color = COLOURS[tracker[5].astype(int)%32]
            tracker_marker = ros_functions.tracker_to_marker(tracker,color,stamp,self.frame_id)
            tracker_marker_list.markers.append(tracker_marker)
        
        # map_based_trackers = [types_helper.lidar2map_coordinates(self.tf_map2lidar,tracker) for tracker in trackers]
        # print("Trackers: ", map_based_trackers)
        if self.DEBUG: print("Types: ", types)
        if self.DEBUG: print("Velocities: ", vels)
        if self.DEBUG: print("MOT markers: ", len(tracker_marker_list.markers))
        self.pub_mot_markers.publish(tracker_marker_list)
        end__ = time.time()
        
        print("Time SmartMOT pipeline: ", end__ - start__)
        # mott2 = time.time()
        # if self.DEBUG: print(f"Time consumed during Multi-Object Tracking pipeline: {mott2-mott1}")
               
        #     #         if self.COMPUTE_MARKERS:    
        #     #             marker = ros_functions.get_detection_marker(detection, is_relevant)
        #     #             map_monitor_filtered_objects.markers.append(marker)
                        
        #     #             if is_relevant:
        #     #                 # OBS: The frequency with which we received these data is not fixed. I.e. 20 steps would not
        #     #                 # necessarily mean 2 s (regarding a frequency of 10 Hz) -> TODO: Fix that
        #     #                 observations_marker, predictions_marker = ros_functions.get_trajectory_marker(self.trackers[detection.object_id][-OBS_LEN:])
        #     #                 map_monitor_filtered_objects.markers.append(observations_marker)
        #     #                 map_monitor_filtered_objects.markers.append(predictions_marker)
                        
        #     # if self.COMPUTE_MARKERS: self.pub_map_monitor_filtered_detections_markers.publish(map_monitor_filtered_objects)
            
        # # print(f"Time consumed: {time.time() - start}")
        
        # #################################################################
        # ################## MOTION PREDICTION PIPELINE ###################
        # #################################################################
        
        # # Preprocess filtered objects as input for the Motion Prediction algorithm

        # if self.PREPROCESS_TRACKERS:
        #     self.state.clear()
        #     id_type = {}
            
        #     # We assume that the ego-vehicle is the first object since we have previously sorted from nearest to furthest
                                
        #     for i in range(len(filtered_objects.gt_3d_object_list)):
        #         filtered_obj = filtered_objects.gt_3d_object_list[i]

        #         # OBS: If a timestep i-th has not been truly observed, that particular observation (x,y,binary_flag) 
        #         # is padded (that is, third dimension set to 0). Otherwise, set to 1
                
        #         # TODO: Is this required? You know the identifier of the ego
                
        #         if filtered_obj.type == "ego_vehicle":
        #             if not "ego" in self.list_of_ids:
        #                 self.list_of_ids["ego"] = [[0, 0, 0] for _ in range(self.motion_predictor.OBS_LEN)] # Initialize buffer

        #             self.list_of_ids["ego"].append([filtered_obj.global_position.x, 
        #                                             filtered_obj.global_position.y,
        #                                             1])
                    
        #             self.state[filtered_obj.object_id] = np.array(self.list_of_ids["ego"][-self.motion_predictor.OBS_LEN:])
        #             id_type[filtered_obj.object_id] = filtered_obj.type
                    
        #         else: # Other agents
        #             adv_id = filtered_obj.object_id
        #             if self.DEBUG: print("Adversary ID: ", adv_id)
        #             x_adv = filtered_obj.global_position.x
        #             y_adv = filtered_obj.global_position.y
                    
        #             if adv_id in self.list_of_ids:
        #                 self.list_of_ids[adv_id].append([x_adv, y_adv, 1])
        #             else:
        #                 self.list_of_ids[adv_id] = [[0, 0, 0] for _ in range(self.motion_predictor.OBS_LEN)]
        #                 self.list_of_ids[adv_id].append([x_adv, y_adv, 1])

        #             self.state[adv_id] = np.array(self.list_of_ids[adv_id][-self.motion_predictor.OBS_LEN:])
        #             id_type[filtered_obj.object_id] = filtered_obj.type
                    
        #             if self.DEBUG: print("Agents state: ", self.state[adv_id])

        #         if (self.timestamp > 0 
        #             and (filtered_obj.object_id in actors_scenario
        #                 or "ego" in actors_scenario)):
        #             if filtered_obj.type == "ego_vehicle":
        #                 agent_to_remove = actors_scenario.index("ego")
        #             else:
        #                 agent_to_remove = actors_scenario.index(filtered_obj.object_id)
        #             actors_scenario.pop(agent_to_remove)

        #     # Set 0,0,0 (padding) for actors that are in the list_of_ids buffer but 
        #     # they have not been observed in the current timestamp

        #     # TODO: Is this correct?
            
        #     # if self.timestamp > 0:
        #     #     for non_observed_actor_id in actors_scenario:
        #     #         self.list_of_ids[non_observed_actor_id].append([0, 0, 0])
                
        #     # Save current observations into .csv to be predicted offline
            
        #     if self.safe_csv:
        #         self.write_csv(self.state, self.timestamp)
                
        #         # if TIME_SCENARIO > 0 and not self.init_stop_callback: 
        #         #     print("AB4COGT: Start collection data")
        #         #     rospy.Timer(rospy.Duration(TIME_SCENARIO), stop_callback)
        #         #     self.init_stop_callback = True
    
        #     # Preprocess trackers
            
        #     valid_agents_info, valid_agents_id = self.motion_predictor.preprocess_trackers(self.state)
                
        #     if valid_agents_info: # Agents with more than a certain number of observations
        #         # Plot observations ROS markers
                
        #         for num_object, valid_agent_info in enumerate(valid_agents_info):
        #             if num_object > 0: # Avoid plotting the ego-vehicle, we already have the URDF marker
        #                 marker = get_observations_marker(valid_agent_info, 
        #                                                     id_type)
        #                 gt_detections_marker_list.markers.append(marker)

        #         self.pub_gt_marker.publish(gt_detections_marker_list)
                
        #     # Online prediction

        #     if self.USE_PREDICTION and valid_agents_info: 
        #         # Predict agents
                
        #         predictions, confidences = self.motion_predictor.predict_agents(valid_agents_info, self.timestamp)

        #         # Plot predictions ROS markers

        #         if len(predictions) > 0:
        #             self.motion_predictor.plot_predictions_ros_markers(predictions, 
        #                                                             confidences, 
        #                                                             valid_agents_id, 
        #                                                             self.ego_vehicle_location.header.stamp,
        #                                                             COLOURS,
        #                                                             apply_colour=APPLY_RANDOM_COLOUR,
        #                                                             lifetime=LIFETIME)  
            
if __name__=="__main__":
    """
    """
    
    node_name = rospy.get_param("/t4ac/perception/tracking_and_prediction/classic/t4ac_smartmot_ros/t4ac_smartmot_ros_node/node_name")
    rospy.init_node(node_name, anonymous=True)
    
    SmartMOT()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ROS Tracking node")
