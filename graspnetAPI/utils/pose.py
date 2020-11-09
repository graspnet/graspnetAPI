__author__ = 'Minghao Gou'
__version__ = '1.0'
"""
define the pose class and functions associated with this class.
"""

import numpy as np
from . import trans3d
from transforms3d.euler import euler2quat

class Pose:
    def __init__(self,id,x,y,z,alpha,beta,gamma):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        # alpha, bata, gamma is in degree
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quat = self.get_quat()
        self.mat_4x4 = self.get_mat_4x4()
        self.translation = self.get_translation()

    def __repr__(self):
        return '\nPose id=%d,x=%f,y=%f,z=%f,alpha=%f,beta=%f,gamma=%f' %(self.id,self.x,self.y,self.z,self.alpha,self.beta,self.gamma)+'\n'+'translation:'+self.translation.__repr__() + '\nquat:'+self.quat.__repr__()+'\nmat_4x4:'+self.mat_4x4.__repr__()

    def get_id(self):
        """
        **Output:**
        
        - return the id of this object
        """
        return self.id

    def get_translation(self):
        """ 
        **Output:**

        - Convert self.x, self.y, self.z into self.translation
        """
        return np.array([self.x,self.y,self.z])

    def get_quat(self):
        """
        **Output:**
        
        - Convert self.alpha, self.beta, self.gamma into self.quat
        """
        euler = np.array([self.alpha, self.beta, self.gamma]) / 180.0 * np.pi
        quat = euler2quat(euler[0],euler[1],euler[2])
        return quat

    def get_mat_4x4(self):
        """
        **Output:**
        
        - Convert self.x, self.y, self.z, self.alpha, self.beta and self.gamma into mat_4x4 pose
        """
        mat_4x4 = trans3d.get_mat(self.x,self.y,self.z,self.alpha,self.beta,self.gamma)
        return mat_4x4

def pose_from_pose_vector(pose_vector):
    """
    **Input:**
    
    - pose_vector: A list in the format of [id,x,y,z,alpha,beta,gamma]
    
    **Output:**
    
    - A pose class instance
    """
    return Pose(id = pose_vector[0],
    x = pose_vector[1],
    y = pose_vector[2],
    z = pose_vector[3],
    alpha = pose_vector[4],
    beta = pose_vector[5],
    gamma = pose_vector[6])

def pose_list_from_pose_vector_list(pose_vector_list):
    """
    **Input:**

    - Pose vector list defined in xmlhandler.py

    **Output:**
    
    - list of poses.
    """
    pose_list = []
    for pose_vector in pose_vector_list:
        pose_list.append(pose_from_pose_vector(pose_vector))
    return pose_list