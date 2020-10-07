import os
from tqdm import tqdm

graspnet_root = '/home/gmh/graspnet'
rect_labels_root = 'rect_labels'

for sceneId in tqdm(range(190), 'Copying Labels'):
    for camera in ['kinect', 'realsense']:
        dest_dir = os.path.join(graspnet_root, 'scenes', 'scene_%04d' % sceneId, camera, 'rect')
        rm_dest_dir = os.path.join(graspnet_root, 'scenes', 'scene_%04d' % sceneId, camera, 'rectangle_grasp')
        os.system('rm -rf {}'.format(rm_dest_dir))
        src_dir = os.path.join(rect_labels_root, 'scene_%04d' % sceneId, camera)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        for annId in range(256):
            src_path = os.path.join(src_dir,'%04d.npy' % annId)
            assert os.path.exists(src_path)
            os.system('cp {} {}'.format(src_path, dest_dir))
