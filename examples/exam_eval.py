__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for evaluate grasps for a scene.
# change the graspnet_root path
import numpy as np
from graspnetAPI import GraspNetEval

####################################################################
graspnet_root = '/DATA2/Benchmark/graspnet' # ROOT PATH FOR GRASPNET
dump_folder = 'dump_full'
####################################################################

if __name__ == '__main__':
    sceneId = 160
    camera = 'kinect'    
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = 'test')
    acc = ge.eval_scene(scene_id = sceneId, camera = camera, dump_folder = dump_folder)
    np_acc = np.array(acc)
    print('mean accuracy:{}'.format(np.mean(np_acc)))
