__author__ = 'mhgou, cxwang and hsfang'
__version__ = '1.0'

import numpy as np
import os
import time
import open3d as o3d

from .graspnet import GraspNet
from .utils.config import get_config
from .utils.eval_utils import get_scene_name, create_table_points, parse_posevector, load_dexnet_model, transform_points, compute_point_distance, compute_closest_points, voxel_sample_points, topk_grasps, get_grasp_score, collision_detection, eval_grasp
from .utils.xmlhandler import xmlReader

class GraspNetEval(GraspNet):
    def __init__(self, root, camera, split):
        super(GraspNetEval, self).__init__(root, camera, split)
        
    def get_scene_models(self, scene_id, ann_id, camera='realsense'):
        '''
            return models in model coordinate
        '''
        model_dir = os.path.join(self.root, 'models')
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), camera, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        model_list = []
        dexmodel_list = []
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list


    def get_model_poses(self, scene_id, ann_id, camera='realsense'):
        '''
            pose_list: object pose from model to camera coordinate
            camera_pose: from camera to world coordinate
            align_mat: from world to table-horizontal coordinate
        '''
        scene_dir = os.path.join(self.root, 'scenes')
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), camera, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), camera, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera, 'annotations', '%04d.xml'% (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat
        
    def eval_scene(self, scene_id, camera, dump_folder):
        model_dir = os.path.join(self.root, 'models')
        dexmodel_dir = os.path.join(self.root, 'models')
        scene_dir = os.path.join(self.root, 'scenes')

        config = get_config()
        table = create_table_points(1.0, 0.05, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.008)
        TOP_K = 50
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        # for scene_id in range(115,116):
        tic = time.time()
        model_list, dexmodel_list, obj_list = self.get_scene_models(scene_id, ann_id=0, camera=camera)
        toc = time.time()
        # print('model loading time: %f' % (toc-tic))
        model_sampled_list = list()
        tic = time.time()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)
        toc = time.time()
        # print('model voxel sample time: %f' % (toc-tic))
        scene_accuracy = []
        for ann_id in range(256):
            print('scene id:{}, ann id:{}'.format(scene_id, ann_id))
            grasps = np.array(np.load(os.path.join(dump_folder,get_scene_name(scene_id), camera, '%04d.npz' % (ann_id,)))['preds'][0])
            obj_list, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id, camera=camera)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # model level list
            tic = time.time()
            grasp_list, score_list, collision_mask_list = eval_grasp(grasps, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008)
            toc = time.time()

            # concat into scene level
            # remove empty
            grasp_list = [x for x in grasp_list if len(x[0])!= 0]
            score_list = [x for x in score_list if len(x)!=0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            # sort in scene level
            grasp_confidence = grasp_list[:,1]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

            #calculate AP
            grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0,TOP_K):
                    # scores[k,fric_idx] is the average score for top k grasps with coefficient of friction at fric
                    if k+1 > len(score_list):
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                    else:
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

            print('Mean Accuracy for grasps under friction_coef 0.2', np.mean(grasp_accuracy[:,0]))
            print('Mean Accuracy for grasps under friction_coef 0.4', np.mean(grasp_accuracy[:,1]))
            print('Mean Accuracy for grasps under friction_coef 0.6', np.mean(grasp_accuracy[:,2]))
            print('Mean Accuracy for grasps under friction_coef 0.8',np.mean(grasp_accuracy[:,3]))
            print('Mean Accuracy for grasps under friction_coef 1.0', np.mean(grasp_accuracy[:,4]))
            print('Mean Accuracy for grasps under friction_coef 1.2',np.mean(grasp_accuracy[:,5]))
            print('Mean Accuracy for',np.mean(grasp_accuracy[:,:]))
            scene_accuracy.append(grasp_accuracy)
        return scene_accuracy