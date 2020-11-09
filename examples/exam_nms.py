__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for grasp nms.
# change the graspnet_root path

####################################################################
graspnet_root = '/home/gmh/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

sceneId = 1
annId = 3

from graspnetAPI import GraspNet
import open3d as o3d
import cv2

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
print('6d grasp:\n{}'.format(_6d_grasp))

# visualize the grasps using open3d
geometries = []
geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = 'kinect'))
geometries += _6d_grasp.random_sample(numGrasp = 1000).to_open3d_geometry_list()
o3d.visualization.draw_geometries(geometries)

nms_grasp = _6d_grasp.nms(translation_thresh = 0.1, rotation_thresh = 30 / 180.0 * 3.1416)
print('grasp after nms:\n{}'.format(nms_grasp))

# visualize the grasps using open3d
geometries = []
geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = 'kinect'))
geometries += nms_grasp.to_open3d_geometry_list()
o3d.visualization.draw_geometries(geometries)