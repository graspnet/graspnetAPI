__author__ = 'mhgou, cxwang and hsfang'
__version__ = '1.0'

import numpy as np
import os
import time
import pickle
import open3d as o3d

from .graspnet import GraspNet
from .grasp import GraspGroup
from .utils.config import get_config
from .utils.eval_utils import get_scene_name, create_table_points, parse_posevector, load_dexnet_model, transform_points, compute_point_distance, compute_closest_points, voxel_sample_points, topk_grasps, get_grasp_score, collision_detection, eval_grasp
from .utils.xmlhandler import xmlReader

class GraspNetEval(GraspNet):
    def __init__(self, root, camera, split = 'test'):
        super(GraspNetEval, self).__init__(root, camera, split)
        
    def get_scene_models(self, scene_id, ann_id):
        '''
            return models in model coordinate
        '''
        model_dir = os.path.join(self.root, 'models')
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        model_list = []
        dexmodel_list = []
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            dex_cache_path = os.path.join(self.root, 'dex_models', '%03d.pkl' % obj_idx)
            if os.path.exists(dex_cache_path):
                with open(dex_cache_path, 'rb') as f:
                    dexmodel = pickle.load(f)
            else:
                dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list


    def get_model_poses(self, scene_id, ann_id):
        '''
            pose_list: object pose from model to camera coordinate
            camera_pose: from camera to world coordinate
            align_mat: from world to table-horizontal coordinate
        '''
        scene_dir = os.path.join(self.root, 'scenes')
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        # print('Scene {}, {}'.format(scene_id, camera))
        scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml'% (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat
        
    def eval_scene(self, scene_id, dump_folder, return_list = False,vis = False):
        config = get_config()
        table = create_table_points(1.0, 0.05, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.008)
        TOP_K = 50
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        # for scene_id in range(115,116):
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)
        # print('model loading time: %f' % (toc-tic))
        model_sampled_list = list()
        # tic = time.time()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)
        # toc = time.time()
        # print('model voxel sample time: %f' % (toc-tic))
        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            # print('scene id:{}, ann id:{}'.format(scene_id, ann_id))
            # grasps = np.array(np.load(os.path.join(dump_folder,get_scene_name(scene_id), camera, '%04d.npz' % (ann_id,)))['preds'][0])
            grasp_group = GraspGroup().from_npy(os.path.join(dump_folder,get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # model level list
            # tic = time.time()
            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008)
            # toc = time.time()

            # concat into scene level
            # remove empty
            grasp_list = [x for x in grasp_list if len(x[0])!= 0]
            score_list = [x for x in score_list if len(x)!=0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            print(f'grasp list:{grasp_list}, len = {len(grasp_list)}')
            print(f'score list:{score_list}, len = {len(score_list)}')
            print(f'collision mask list:{collision_mask_list}, len = {len(collision_mask_list)}')
            if vis:
                gg = GraspGroup(grasp_list)
                scores = np.array(score_list)
                scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.set_scores(scores)
                gg.set_widths(0.1 * np.ones((len(gg)), dtype = np.float32))
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)
                # draw_geometries
                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)
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

            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[0]), np.mean(grasp_accuracy[:,0])) # 0.1
            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[1]), np.mean(grasp_accuracy[:,1])) # 0.2
            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[2]), np.mean(grasp_accuracy[:,2])) # 0.3
            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[4]), np.mean(grasp_accuracy[:,4])) # 0.5
            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[6]), np.mean(grasp_accuracy[:,6])) # 0.7
            # print('Mean Accuracy for grasps under friction_coef {}'.format(list_coe_of_friction[8]), np.mean(grasp_accuracy[:,8])) # 0.9
            print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id),np.mean(grasp_accuracy[:,:]), end='')
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def parallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for scene_id in scene_ids:
            res_list.append(p.apply_async(self.eval_scene, (scene_id, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_seen(self, dump_folder, proc = 2):
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(100, 130)), dump_folder = dump_folder, proc = proc))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}'.format(self.camera, ap, ap))
        return res, ap

    def eval_all(self, dump_folder, proc = 2):
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(100, 190)), dump_folder = dump_folder, proc = proc))
        ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(self.camera, ap[0], ap[1], ap[2], ap[3]))
        return res, ap
