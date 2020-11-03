__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for generating rectangle grasp from 6d grasp.
# change the graspnet_root path and NUM_PROCESS

from graspnetAPI import GraspNet
from graspnetAPI.graspnet import TOTAL_SCENE_NUM
import os
import numpy as np
from tqdm import tqdm

######################################################################
NUM_PROCESS = 24 # change NUM_PROCESS to the number of cores to use. #
######################################################################

def generate_scene_rectangle_grasp(sceneId, dump_folder, camera):
    g = GraspNet(graspnet_root, camera=camera, split='all')
    objIds = g.getObjIds(sceneIds = sceneId)
    grasp_labels = g.loadGraspLabels(objIds)
    collision_labels = g.loadCollisionLabels(sceneIds = sceneId)
    scene_dir = os.path.join(dump_folder,'scene_%04d' % sceneId)
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)
    camera_dir = os.path.join(scene_dir, camera)
    if not os.path.exists(camera_dir):
        os.mkdir(camera_dir)
    for annId in tqdm(range(256), 'Scene:{}, Camera:{}'.format(sceneId, camera)):
        _6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, grasp_labels = grasp_labels, collision_labels = collision_labels, fric_coef_thresh = 1.0)
        rect_grasp_group = _6d_grasp.to_rect_grasp_group(camera)
        rect_grasp_group.save_npy(os.path.join(camera_dir, '%04d.npy' % annId))

if __name__ == '__main__':
    ####################################################################
    graspnet_root = '/home/minghao/graspnet' # ROOT PATH FOR GRASPNET ##
    ####################################################################

    dump_folder = 'rect_labels'
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)

    if NUM_PROCESS > 1:
        from multiprocessing import Pool
        pool = Pool(24)
        for camera in ['realsense', 'kinect']:
            for sceneId in range(120):
                pool.apply_async(func = generate_scene_rectangle_grasp, args = (sceneId, dump_folder, camera))
        pool.close()
        pool.join()
    
    else:
        generate_scene_rectangle_grasp(sceneId, dump_folder, camera)
