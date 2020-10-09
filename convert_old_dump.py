import numpy as np
import open3d as o3d

from graspnetAPI import GraspNet
from graspnetAPI import GraspGroup
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix


def convert_dump(src_path, dst_path):
    '''
    **Input:**

    - src_path: string of the old dump npy file path.

    - dst_path: string of the new grasp group npy file path.

    **Output:**

    - No output but transforms the old dump to new format.
    '''
    # // 0: graspable score
    # // 1: graspscore
    # // 2-4: center
    # // 5-7: approaching vector
    # // 8: angle 
    # // 9-10: depth, width
    # // height is not given which is a constant of 2
    dump = np.load(src_path)
    preds = dump['preds'][0]
    num = preds.shape[0]
    assert num == 1024, 'num should be 1024'
    scores = preds[:,1].reshape((-1, 1))
    translations = preds[:,2:5]
    app_vectors = preds[:,5:8]
    angles = preds[:,8]
    depths = preds[:,9].reshape((-1, 1))
    widths = preds[:,10].reshape((-1, 1))
    heights = 0.02 * np.ones((num)).reshape((-1, 1))
    object_ids = -1 * np.ones((num)).reshape((-1, 1))
    
    rotations = batch_viewpoint_params_to_matrix(app_vectors, angles).reshape(-1, 9)
    angles = angles.reshape((-1, 1))
    print(f'scores:{scores}, widths:{widths}, heights:{heights}, depths:{depths}, tranlations:{translations}, rotations:{rotations}, object_ids:{object_ids}')
    grasp_group = GraspGroup(np.hstack((scores, widths, heights, depths, rotations, translations, object_ids)))
    grasp_group.save_npy(dst_path)

if __name__ == '__main__':
    src_path = '0000.npz'
    dst_path = 's100_k_0.npy'
    convert_dump(src_path, dst_path)

    # ####################################################################
    # graspnet_root = '/home/gmh/graspnet' # ROOT PATH FOR GRASPNET
    # ####################################################################
    # g = GraspNet(graspnet_root, camera='kinect', split='all')

    # _6d_grasp = GraspGroup()
    # _6d_grasp.from_npy('s100_k_0.npy')
    # geometries = []
    # geometries.append(g.loadScenePointCloud(sceneId = 100, annId = 0, camera = 'kinect'))
    # geometries += _6d_grasp.random_sample(numGrasp = 20).to_open3d_geometry_list()
    # o3d.visualization.draw_geometries(geometries)