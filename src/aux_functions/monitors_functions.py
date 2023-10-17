#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on tue Nov 07 11:56:48 2021
@author: Carlos Gómez Huélamo
"""

import math
import numpy as np
import copy
import tf
import sys

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import geometry_msgs.msg
import visualization_msgs.msg
import nav_msgs.msg
import t4ac_msgs.msg

from . import geometric_functions

monitorized_area_gap = 20

# Auxiliar functions

def apply_tf(source_location, transform):  
    """
    Input: t4ac_msgs.msg.Node() in the source frame
    Output: t4ac_msgs.msg.Node() in the target frame 
    """
    centroid = np.array([0.0,0.0,0.0,1.0]).reshape(4,1)

    try:
        centroid[0,0] = source_location.x 
        centroid[1,0] = source_location.y
        centroid[2,0] = source_location.z
    except:
        centroid[0,0] = source_location[0] # LiDAR points (3,)
        centroid[1,0] = source_location[1]
        centroid[2,0] = source_location[2]

    aux = np.dot(transform,centroid) 

    target_location = t4ac_msgs.msg.Node()
    target_location.x = aux[0,0]
    target_location.y = aux[1,0]
    target_location.z = aux[2,0]

    return target_location

def node_to_array(node):
    """
    t4ac_msgs.msg.Node (x,y,z) -> array 
    """

    aux = np.array([node.x,node.y,node.z])
    return aux

# Inside route functions

def inside_polygon(p, polygon):
    """
    This functions checks if a point is inside a certain lane (or area), a.k.a. polygon.
    (https://jsbsan.blogspot.com/2011/01/saber-si-un-punto-esta-dentro-o-fuera.html)
    Takes a point and a polygon and returns if it is inside (1) or outside(0).
    """
    counter = 0
    xinters = 0

    p1 = t4ac_msgs.msg.Node()
    p2 = t4ac_msgs.msg.Node()

    p1 = polygon[0]

    for i in range(1,len(polygon)+1):
        p2 = polygon[i%len(polygon)]

        if (p.y > min(p1.y,p2.y)):
            if (p.y <= max(p1.y,p2.y)):
                if (p.x <= max(p1.x,p2.x)):
                    if (p1.y != p2.y):
                        xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x
                        if (p1.x == p2.x or p.x <= xinters):
                            counter += 1
        p1 = p2

    if (counter % 2 == 0):
        return False
    else:
        return True

def find_closest_segment(way, point):
    """
    This functions obtains the closest segment (two nodes) of a certain way (left or right)
    Returns the nodes that comprise this segment and the distance
    """

    min_distance = 999999

    closest_node_0 = t4ac_msgs.msg.Node()
    closest_node_1 = t4ac_msgs.msg.Node()

    closest = -1

    for i in range(len(way)-1):
        node_0 = t4ac_msgs.msg.Node()
        node_0.x = way[i].x 
        node_0.y = way[i].y

        node_1 = t4ac_msgs.msg.Node()
        node_1.x = way[i+1].x 
        node_1.y = way[i+1].y
        try:
            distance, _ =  geometric_functions.pnt2line(point, node_0, node_1)
        except:
            import pdb
            pdb.set_trace()

        # TODO: check repeated points on route
        if distance == -1:
            return -1, -1, -1 
        if (distance < min_distance):
            min_distance = distance
            closest = i
            closest_node_0 = node_0
            closest_node_1 = node_1

    if min_distance != 999999:   
        if (closest > 0) and (closest < len(way)-2):
            # return closest-1,closest+2,min_distance
            return closest,closest+1,min_distance
        elif (closest > 0) and (closest == len(way)-2):
            # return closest-1,closest+1,min_distance
            return closest,closest+1,min_distance
        elif closest == 0 and (len(way) == 2): 
            return 0,1,min_distance
        elif closest == 0 and (len(way) > 2): 
            return 0,2,min_distance
    else:
        return -1, -1, -1

def inside_lane(point, lane, object_type=None):
    """
    """

    try:
        assert not np.isnan(point.x)
    except: # point is not a t4ac_msgs.msg.Node object
         point = copy.copy(point)
         point = point[0]

    fn_l = lane.left.way[0] # First node left
    fn_r = lane.right.way[0] # First node right
    lane_width = math.sqrt(pow(fn_l.x-fn_r.x,2)+pow(fn_l.y-fn_r.y,2))

    n0l_index, n1l_index, dist2segment_left = find_closest_segment(lane.left.way, point)
    n0_left = lane.left.way[n0l_index]
    n1_left = lane.left.way[n1l_index]
    try:
        dist2segment_left_v2, _ = geometric_functions.pnt2line(point, n0_left, n1_left)
    except:
        import pdb
        pdb.set_trace()
    n0_right = lane.right.way[n0l_index]
    n1_right = lane.right.way[n1l_index]
    
    try:
        dist2segment_right, _ = geometric_functions.pnt2line(point, n0_right, n1_right)
    except:
        import pdb
        pdb.set_trace()

    # TODO: check repeated points on route
    if dist2segment_right == -1:
        return False, False, [], -1, -1
    
    if ((dist2segment_left <= lane_width + monitorized_area_gap) or (dist2segment_right <= lane_width + monitorized_area_gap)):
        n0_left = lane.left.way[n0l_index]
        n1_left = lane.left.way[n1l_index]

        test = lambda dist2segment_left,dist2segment_right : (dist2segment_left,-1) if(dist2segment_left <= dist2segment_right) else (dist2segment_right,1)
        nearest_distance,role = test(dist2segment_left,dist2segment_right)

        # Role = -1 -> Closer to the left way or in the middle
        # Role =  1 -> Closer to the right way
        
        if ((object_type == "person" or object_type == "bicycle") and lane.role == "front_current_lane"):
            if (dist2segment_left > lane_width or dist2segment_right > lane_width): # Outside the corresponding section

                if role == -1: 
                    polygon_width = monitorized_area_gap + lane_width
                    d,e = geometric_functions.calculate_rectangle(n0_right, n1_right, point, polygon_width)
                    n0_aux = t4ac_msgs.msg.Node(d[0],d[1],0)
                    n1_aux = t4ac_msgs.msg.Node(e[0],e[1],0)
                    polygon = [n0_aux,n1_aux,n1_right,n0_right]
                    return False, False, [], -1, -1 # TODO: IMPROVE THIS. DO NOT CONSIDER "PERSON" IN THE ADYACENT LEFT LANE, 
                                                    #       ONLY UNEXPECTED COMING FROM THE RIGHT
                else: # Closer to the right way
                    polygon_width = monitorized_area_gap + lane_width
                    d,e = geometric_functions.calculate_rectangle(n0_left, n1_left, point, polygon_width)
                    n0_aux = t4ac_msgs.msg.Node(d[0],d[1],0)
                    n1_aux = t4ac_msgs.msg.Node(e[0],e[1],0)

                    polygon = [n0_left,n1_left,n1_aux,n0_aux]
            else: # Inside the original polygon
                polygon = [n0_left,n1_left,n1_right,n0_right]
        else:
            polygon = [n0_left,n1_left,n1_right,n0_right]
        
        road = [n0_left,n1_left,n1_right,n0_right]

        is_relevant = inside_polygon(point, polygon)
        in_road = inside_polygon(point, road)

        segment_index = n0l_index

        # print("Is relevant: ", is_relevant)
        # print("Is inside road: ", in_road)
        # print("Nearest distance: ", nearest_distance)
        # print("Point: ", point.x, point.y)
        # for pt in polygon:
        #     print("x y: ", pt.x, pt.y)
      
        return is_relevant, in_road, polygon, nearest_distance, segment_index
    else:
        return False, False, [], -1, -1