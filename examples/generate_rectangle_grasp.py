__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for loading grasp for a scene.
# change the graspnet_root path

from graspnetAPI import GraspNet
from graspnetAPI.graspnet import TOTAL_SCENE_NUM
import os
import numpy as np
from tqdm import tqdm



def generate_scene_rectangle_grasp(sceneId, dump_folder, camera, graspnet = None, grasp_labels = None, collision_labels = None):
    if graspnet is None:
        graspnet = GraspNet(graspnet_root, camera=camera, split='all')
    scene_dir = os.path.join(dump_folder,'scene_%04d' % sceneId)
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)
    camera_dir = os.path.join(scene_dir, camera)
    if not os.path.exists(camera_dir):
        os.mkdir(camera_dir)
    for annId in tqdm(range(256), 'Scene:{}, Camera:{}'.format(sceneId, camera)):
        _6d_grasp = graspnet.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, grasp_labels = grasp_labels, collision_labels = collision_labels, fric_coef_thresh = 1.0)
        rect_grasp_group = _6d_grasp.to_rect_grasp_group(camera)
        rect_grasp_group.save_npy(os.path.join(camera_dir, '%04d.npy' % annId))

if __name__ == '__main__':
    ####################################################################
    graspnet_root = '/home/gmh/graspnet'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    dump_folder = 'rect_labels'
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)

    for camera in ['kinect', 'kinect']:
        # initialize a GraspNet instance
        for sceneId in range(TOTAL_SCENE_NUM):
            g = GraspNet(graspnet_root, camera=camera, split='all')
            objIds = g.getObjIds(sceneIds = sceneId)
            grasp_labels = g.loadGraspLabels(objIds)
            collision_labels = g.loadCollisionLabels(sceneIds = sceneId)
            generate_scene_rectangle_grasp(sceneId = sceneId, dump_folder = dump_folder, camera = camera, graspnet = g, grasp_labels = grasp_labels, collision_labels = collision_labels)

