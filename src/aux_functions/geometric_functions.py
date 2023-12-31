#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 17:36:17 2020
@author: Carlos Gómez Huélamo
"""

from re import A
import numpy as np
import math

from shapely.geometry import Polygon

def compute_corners(bbox):
    """
    Compute the corners of the rectangle given its x,y centroid, width and length and
    its rotation (clockwise if you see the screen, according to OpenCV)
    """
    
    rotation = bbox[4]
    
    if rotation > math.pi:
        rotation = rotation - math.pi
    
    R = rotz(rotation)

    # 3D bounding box corners

    x, y, w, l = bbox[0], bbox[1], bbox[2], bbox[3]

    x_corners = [-l/2,-l/2,l/2,l/2]
    y_corners = [w/2,-w/2,w/2,-w/2]
    z_corners = [0,0,0,0]

    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d = np.dot(R, corners_3d)[0:2]
    corners_3d = corners_3d + np.vstack([x,y])
    
    corners = []
    for i in range(4):
        # c = int(round(corners_3d[0,i].item())), int(round(corners_3d[1,i].item()))
        c = corners_3d[0,i], corners_3d[1,i]
        corners.append(c)
    corners = tuple(corners)
    return corners

def iou(bb_1,bb_2): 
  """
  Computes IOU between two (possibly) rotated bounding boxes in the form [x,y,w,l,theta]
  """

  corners_1 = compute_corners(bb_1)
  corners_2 = compute_corners(bb_2)

#   print("Corners 1: ", corners_1)
#   print("Corners 2: ", corners_2)

  # To build the polygon -> Left-bottom corner, Right-bottom, Top-right corner, Top-left corner

  b1 = Polygon([corners_1[2],corners_1[3],corners_1[1],corners_1[0]]) 
  b2 = Polygon([corners_2[2],corners_2[3],corners_2[1],corners_2[0]])

#   print("b1: ", b1)
#   print("b2: ", b2)
  
  intersection = b1.intersection(b2).area
  union = b1.union(b2).area

#   print("I U: ", intersection, union)

  if union > 0.0:
    o = intersection / union
    return(o)
  else:
    return 0.0  
    
def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    try:
        x,y,z = b.x, b.y, 0.0
        X,Y,Z = e.x, e.y, 0.0
    except:
        x,y,z = b
        X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v

    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    try:   
        X,Y,Z = w.x, w.y, 0.0
    except:
        X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)

    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def calculate_rectangle(node_0, node_1, point, polygon_width):
    """
    """

    a = (node_0.x, node_0.y)
    b = (node_1.x, node_1.y)
    c = (point.x, point.y)

    # Distance from dot C to line AB (https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points)
    
    # distance = abs((b[1]-a[1])*c[0] - (b[0]-a[0])*c[1] + b[0]*a[1] - b[1]*a[0] ) / math.sqrt((b[1]-a[1])**2 + (b[0]-a[0])**2)
    distance = polygon_width # Create a rectangle of lane_width + monitorized_area_gap instead of only dist point to segment

    # Let AB be the vector from A to B

    ab=[b[0]-a[0],b[1]-a[1]]

    # Unit vector perpendicular to AB (https://en.wikipedia.org/wiki/Unit_vector)

    perpendicularSize = math.sqrt(ab[0]**2+ab[1]**2)
    unit = [-ab[1]/perpendicularSize ,ab[0]/perpendicularSize]

    # Multiply that unit vector by the distance from C to AB to get the vectors for the "sides" of your rectangle
    sideVector = [unit[0]*distance,unit[1]*distance]

    # Create new points D and E by offsetting A and B by the vector for the sides of the rectangle

    # N.B. The perpendicular vector to a segment AB, upwards and downwards, so, in order to know the correct
    # values of the additional side points, first calculate both points with +/+ and then -/-, and calculate 
    # the closest middle point to the point of interest. Then, take the corresponding points to complete
    # your rectangle

    d1=[a[0]+sideVector[0],a[1]+sideVector[1]]
    e1=[b[0]+sideVector[0],b[1]+sideVector[1]]

    mid_point_1 = np.add(d1,e1)/2
    dist1 = np.linalg.norm(mid_point_1-np.array(c))

    d2=[a[0]-sideVector[0],a[1]-sideVector[1]]
    e2=[b[0]-sideVector[0],b[1]-sideVector[1]]

    mid_point_2 = np.add(d2,e2)/2
    dist2 = np.linalg.norm(mid_point_2-np.array(c))

    if dist1 < dist2:
        return d1,e1
    else:
        return d2,e2

def rotz(t):
    """ 
    Rotation about the z-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,  0],
                     [0,   0,  1]])