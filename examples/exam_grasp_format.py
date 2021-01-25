__author__ = 'mhgou'
__version__ = '1.0'

from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2
import numpy as np

# GraspNetAPI example for loading grasp for a scene.
# change the graspnet_root path

####################################################################
graspnet_root = '/disk1/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

sceneId = 1
annId = 3

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
print('6d grasp:\n{}'.format(_6d_grasp))

# _6d_grasp is an GraspGroup instance defined in grasp.py
print('_6d_grasp:\n{}'.format(_6d_grasp))

# index
grasp = _6d_grasp[0]
print('_6d_grasp[0](grasp):\n{}'.format(grasp))

# slice
print('_6d_grasp[0:2]:\n{}'.format(_6d_grasp[0:2]))
print('_6d_grasp[[0,1]]:\n{}'.format(_6d_grasp[[0,1]]))

# grasp is a Grasp instance defined in grasp.py
# access and set properties
print('grasp.translation={}'.format(grasp.translation))
grasp.translation = np.array([1.0, 2.0, 3.0])
print('After modification, grasp.translation={}'.format(grasp.translation))
print('grasp.rotation_matrix={}'.format(grasp.rotation_matrix))
grasp.rotation_matrix = np.eye(3).reshape((9))
print('After modification, grasp.rotation_matrix={}'.format(grasp.rotation_matrix))
print('grasp.width={}, height:{}, depth:{}, score:{}'.format(grasp.width, grasp.height, grasp.depth, grasp.score))
print('More operation on Grasp and GraspGroup can be seen in the API document')


# load rectangle grasps of scene 1 with annotation id = 3, camera = realsense and fric_coef_thresh = 0.2
rect_grasp_group = g.loadGrasp(sceneId = sceneId, annId = annId, format = 'rect', camera = 'realsense', fric_coef_thresh = 0.2)
print('rectangle grasp group:\n{}'.format(rect_grasp_group))

# rect_grasp is an RectGraspGroup instance defined in grasp.py
print('rect_grasp_group:\n{}'.format(rect_grasp_group))

# index
rect_grasp = rect_grasp_group[0]
print('rect_grasp_group[0](rect_grasp):\n{}'.format(rect_grasp))

# slice
print('rect_grasp_group[0:2]:\n{}'.format(rect_grasp_group[0:2]))
print('rect_grasp_group[[0,1]]:\n{}'.format(rect_grasp_group[[0,1]]))

# properties of rect_grasp 
print('rect_grasp.center_point:{}, open_point:{}, height:{}, score:{}'.format(rect_grasp.center_point, rect_grasp.open_point, rect_grasp.height, rect_grasp.score))

# transform grasp
g = Grasp() # simple Grasp
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

# Grasp before transformation
o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])
g.translation = np.array((0,0,0.01))

# setup a transformation matrix
T = np.eye(4)
T[:3,3] = np.array((0.01, 0.02, 0.03))
T[:3,:3] = np.array([[0,0,1.0],[1,0,0],[0,1,0]])
g.transform(T)

# Grasp after transformation
o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])

g1 = Grasp()
gg = GraspGroup()
gg.add(g)
gg.add(g1)

# GraspGroup before transformation
o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])

gg.transform(T)

# GraspGroup after transformation
o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])