# #!/usr/bin/env python3.8
# # -*- coding: utf-8 -*-

# """
# Created on Thu May  7 17:38:44 2020

# @author: Carlos Gómez-Huélamo

# SmartMOT: Code to track the detections given by a sensor fusion algorithm (converted into Bird's Eye View (Image frame, z-axis inwards 
# with the origin located at the top-left corner) using the SORT (Simple Online and Real-Time Tracking) algorithm as backbone,
# and filtered by a Monitored Lanes-based Attention module

# Communications are based on ROS (Robot Operating Sytem)

# Inputs: 3D Object Detections topic
# Outputs: Tracked obstacles topic and monitors information (collision prediction, front obstacle, etc.)

# Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
# in order to make easier the decision making processes.
# """

# from __future__ import print_function

# # General-use imports

# import os
# import sys
# import git
# import time
# import cv2
# import numpy as np
# import math
# import matplotlib
# matplotlib.use('Agg') # In order to avoid: RuntimeError: main thread is not in main loop exception

# from argparse import ArgumentParser

# # Sklearn imports

# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures

# # Custom functions imports

# repo = git.Repo(__file__, search_parent_directories=True)
# BASE_DIR = repo.working_tree_dir
# sys.path.append(BASE_DIR)

# from src.aux_functions import geometric_functions
# from src.aux_functions import monitors_functions
# from src.aux_functions import sort_functions
# from src.aux_functions import tracking_functions
# from src.aux_functions import ros_functions

# # ROS imports

# import rospy
# import visualization_msgs.msg
# import sensor_msgs.msg
# import nav_msgs.msg
# import std_msgs.msg
# import t4ac_msgs.msg
# import tf 

# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from t4ac_msgs.msg import BEV_detection, BEV_detections_list, MonitorizedLanes, Node, Obstacle
# from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer, Subscriber

# # Auxiliar variables

# ## Tracking and Prediction

# detection_threshold = 0.3
# max_age = 3
# min_hits = 1
# seconds_ahead = 3 # Predict all trajectories at least M seconds ahead
# num_steps = 4 # Predict N bounding boxes (steps) ahead regarding M seconds ahead  

# ## KITTI evaluation

# kitti = 0 
# filename = ""
# path = os.path.curdir + "/results" + filename

# header_synchro = 100
# slop = 0.5

# class SmartMOT:
#     def __init__(self):
#         # Auxiliar variables
        
#         self.COMPUTE_MARKERS = True
        
#         # Transforms
        
#         self.tf_map2lidar = np.zeros((4,4))
        
#         # Arguments from ROS params

#         root = rospy.get_param('/t4ac/perception/tracking_and_prediction/classic/t4ac_smartmot_ros/t4ac_smartmot_ros_node/root')

#         self.display = rospy.get_param(os.path.join(root,"display"))
#         self.trajectory_prediction = rospy.get_param(os.path.join(root,"trajectory_prediction"))
#         self.ros = rospy.get_param(os.path.join(root,"use_ros"))
#         self.grid = rospy.get_param(os.path.join(root,"use_grid"))
#         self.map_frame = rospy.get_param('t4ac/frames/map')
#         self.lidar_frame = rospy.get_param('t4ac/frames/lidar')
        
#         # ROS communications
        
#         ## ROS publishers
        
#         map_monitor_filtered_detections_markers = "/t4ac/perception/detection/map_monitor_filtered_detections"
#         self.pub_map_monitor_filtered_detections_markers = rospy.Publisher(map_monitor_filtered_detections_markers, 
#                                                                            visualization_msgs.msg.MarkerArray, 
#                                                                            queue_size=10)
        
#         ## ROS subscribers

#         detections_topic = rospy.get_param(os.path.join(root,"sub_BEV_merged_obstacles"))
#         location_topic = rospy.get_param(os.path.join(root,"sub_localization_pose"))
#         monitorized_lanes_topic = rospy.get_param(os.path.join(root,"sub_monitorized_lanes"))

#         self.sub_detections = Subscriber(detections_topic, BEV_detections_list)
#         self.sub_ego_location = Subscriber(location_topic, nav_msgs.msg.Odometry)
#         self.sub_monitored_lanes = Subscriber(monitorized_lanes_topic, MonitorizedLanes)

#         self.ts = ApproximateTimeSynchronizer([self.sub_detections, 
#                                                self.sub_ego_location, 
#                                                self.sub_monitored_lanes], 
#                                                header_synchro, slop)
#         self.ts.registerCallback(self.smartmot_callback)
        
#         self.listener = tf.TransformListener()
        
#     def smartmot_callback(self, detections_msg, ego_location_msg, monitored_lanes_msg):
#         """
#         """
        
#         # Lookup transform LiDAR2Map
#         try:                                                         # Target        # Source
#             (translation,quaternion) = self.listener.lookupTransform(self.map_frame, self.lidar_frame, rospy.Time(0)) 
#             # rospy.Time(0) get us the latest available transform
#             rot_matrix = tf.transformations.quaternion_matrix(quaternion)
            
#             self.tf_map2lidar = rot_matrix
#             self.tf_map2lidar[:3,3] = self.tf_map2lidar[:3,3] + translation # This matrix transforms local to global coordinates

#         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#             print("\033[1;33m"+"TF Map2LiDAR exception in SmartMOT ROS node"+'\033[0;m')
#             self.ts.registerCallback(self.smartmot_callback)
            
#         # Filter objects (relevant detections to be tracked are in the monitored area)
        
#         self.trackers = {}
        
#         if np.any(self.tf_map2lidar): # Check if TF is initialized
#             orientation_q = ego_location_msg.pose.pose.orientation
#             orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
#             (ego_roll, ego_pitch, ego_yaw) = euler_from_quaternion(orientation_list)
                        
#             map_monitor_filtered_objects = visualization_msgs.msg.MarkerArray()

#             for detection_index,detection in enumerate(detections_msg.bev_detections_list):
#                 if detection_index > 0: # We assume the first object, if using GT, it is the ego-vehicle. Avoid this.
#                     detection.o = ego_yaw - detection.o # To obtain the angle in global frame
                    
#                     is_relevant = False
                    
#                     # Transform detection to global coordinates
                    
#                     obstacle_local_position = t4ac_msgs.msg.Node(detection.x, # LiDAR coordinates
#                                                                  detection.y,
#                                                                  0)
#                     obstacle_global_position = monitors_functions.apply_tf(obstacle_local_position,
#                                                                            self.tf_map2lidar)  
                                      
#                     for lane in monitored_lanes_msg.lanes:
#                         if len(lane.left.way)> 1:
#                             is_relevant, in_road, particular_monitorized_area, _, _ = monitors_functions.inside_lane(obstacle_global_position, 
#                                                                                                                      lane, 
#                                                                                                                      detection.type)
                        
#                             if is_relevant:
#                                 if detection.object_id not in self.trackers.keys():
#                                     # self.trackers[detection.object_id] = [detection]
#                                     self.trackers[detection_index] = [detection]
#                                 else:
#                                     self.trackers[detection.object_id].append(detection)
#                                 break
                    
#                     if self.COMPUTE_MARKERS:    
#                         marker = ros_functions.get_detection_marker(detection, 
#                                                                     obstacle_global_position, 
#                                                                     is_relevant)
#                         map_monitor_filtered_objects.markers.append(marker)
                        
#                         # if is_relevant:
#                         #     # OBS: The frequency with which we received these data is not fixed. I.e. 20 steps would not
#                         #     # necessarily mean 2 s (regarding a frequency of 10 Hz) -> TODO: Fix that
#                         #     # observations_marker, predictions_marker = ros_functions.get_trajectory_marker(self.trackers[detection.object_id][-OBS_LEN:])
#                         #     # observations_marker, predictions_marker = ros_functions.get_trajectory_marker(self.trackers[detection.object_id])
#                         #     observations_marker, predictions_marker = ros_functions.get_trajectory_marker(self.trackers[detection_index])
#                         #     map_monitor_filtered_objects.markers.append(observations_marker)
#                         #     map_monitor_filtered_objects.markers.append(predictions_marker)
            
#             if self.COMPUTE_MARKERS: self.pub_map_monitor_filtered_detections_markers.publish(map_monitor_filtered_objects)
            
# def main() -> None:
#     node_name = rospy.get_param("/t4ac/perception/tracking_and_prediction/classic/t4ac_smartmot_ros/t4ac_smartmot_ros_node/node_name")
#     rospy.init_node(node_name, anonymous=True)
    
#     SmartMOT()

#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         rospy.loginfo("Shutting down ROS Tracking node")

# if __name__ == "__main__":
#     main()