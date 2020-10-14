__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for evaluate grasps for a scene.
# change the graspnet_root path
import numpy as np
from graspnetAPI import GraspNetEval

####################################################################
# graspnet_root = '/home/gmh/graspnet' # ROOT PATH FOR GRASPNET
graspnet_root = '/home/minghao/graspnet' # ROOT PATH FOR GRASPNET
# dump_folder = '/home/minghao/hdd/dump_new'
dump_folder = '/home/minghao/ssd/multi_dump/new_grasp-0'
####################################################################

if __name__ == '__main__':
    sceneId = 160
    camera = 'kinect'    
    ge_k = GraspNetEval(root = graspnet_root, camera = 'kinect', split = 'test')
    ge_r = GraspNetEval(root = graspnet_root, camera = 'realsense', split = 'test')
    
    print('Evaluating scene:{}, camera:{}'.format(sceneId, camera))
    acc = ge_k.eval_scene(scene_id = sceneId, dump_folder = dump_folder)
    np_acc = np.array(acc)
    print('mean accuracy:{}'.format(np.mean(np_acc)))

    print('Evaluating kinect')
    res, ap = ge_k.eval_all(dump_folder, proc = 24)
    print('Evaluating realsense')
    res, ap = ge_r.eval_all(dump_folder, proc = 24)
