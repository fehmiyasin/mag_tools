"""
This python module stores some functions that are useful for vector
transformations. These were developed through research work on magnetism"""
__author__ = "Fehmi Yasin"
__date__ = "20/03/13"
__version__ = "1.0"
__maintainer__ = "Fehmi Yasin"
__email__ = "fehmi.yasin@riken.jp"

import numpy as np

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis. Theta must be in radians"""
    THETA=-theta
    R = np.array([[1,0,0],[0,np.cos(THETA),-np.sin(THETA)],[0, np.sin(THETA), np.cos(THETA)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    """
    Rotates 3-D vector around y-axis. Theta must be in radians.
    Note that THETA must be negative to that input in x_rotation and z_rotation
    in order to match the sign convention of scipy.ndimage.rotate().
    """
    THETA=theta
    R = np.array([[np.cos(THETA),0,np.sin(THETA)],[0,1,0],[-np.sin(THETA), 0, np.cos(THETA)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis. Theta must be in radians"""
    THETA=-theta
    R = np.array([[np.cos(THETA), -np.sin(THETA),0],[np.sin(THETA), np.cos(THETA),0],[0,0,1]])
    return np.dot(R,vector)

def x_rot_2D(vec_x,vec_y,vec_z,theta):                                  
    """Rotates an array of vector components around x-axis of vector space
       (not array). 
    """                 
    vec_x_rot=np.zeros_like(vec_x,dtype=np.float64)                                      
    vec_y_rot=np.zeros_like(vec_y,dtype=np.float64)                                      
    vec_z_rot=np.zeros_like(vec_z,dtype=np.float64)
    for j,_ in enumerate(vec_x): 
        for i,_ in enumerate(vec_x[0]):
            vector=np.array([vec_x[j,i],vec_y[j,i],vec_z[j,i]])
            vec_rot=x_rotation(vector,theta)
            vec_x_rot[j,i]=np.round(vec_rot[0],3)
            vec_y_rot[j,i]=np.round(vec_rot[1],3)
            vec_z_rot[j,i]=np.round(vec_rot[2],3)
    return vec_x_rot,vec_y_rot,vec_z_rot

def y_rot_2D(vec_x,vec_y,vec_z,theta):
    """Rotates an array of vector components around x-axis of vector space
       (not array).
    """
    vec_x_rot=np.zeros_like(vec_x,dtype=np.float64)                                      
    vec_y_rot=np.zeros_like(vec_y,dtype=np.float64)                                      
    vec_z_rot=np.zeros_like(vec_z,dtype=np.float64)
    for j,_ in enumerate(vec_x): 
        for i,_ in enumerate(vec_x[0]):
            vector=np.array([vec_x[j,i],vec_y[j,i],vec_z[j,i]])
            vec_rot=y_rotation(vector,theta)
            vec_x_rot[j,i]=np.round(vec_rot[0],3)
            vec_y_rot[j,i]=np.round(vec_rot[1],3)
            vec_z_rot[j,i]=np.round(vec_rot[2],3)
    return vec_x_rot,vec_y_rot,vec_z_rot

def z_rot_2D(vec_x,vec_y,vec_z,theta):
    """Rotates an array of vector components around z-axis of vector space (not array)"""
    vec_x_rot=np.zeros_like(vec_x,dtype=np.float64)                                      
    vec_y_rot=np.zeros_like(vec_y,dtype=np.float64)                                      
    vec_z_rot=np.zeros_like(vec_z,dtype=np.float64)
    for j,_ in enumerate(vec_x): 
        for i,_ in enumerate(vec_x[0]):
            vector=np.array([vec_x[j,i],vec_y[j,i],vec_z[j,i]])
            vec_rot=z_rotation(vector,theta)
            vec_x_rot[j,i]=np.round(vec_rot[0],3)
            vec_y_rot[j,i]=np.round(vec_rot[1],3)
            vec_z_rot[j,i]=np.round(vec_rot[2],3)
    return vec_x_rot,vec_y_rot,vec_z_rot
