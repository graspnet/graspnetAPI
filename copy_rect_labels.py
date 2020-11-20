import os
from tqdm import tqdm

### change the root to you path #### 
graspnet_root = '/home/gmh/graspnet'

### change the root to the folder contains rectangle grasp labels ###
rect_labels_root = 'rect_labels'

for sceneId in tqdm(range(190), 'Copying Rectangle Grasp Labels'):
    for camera in ['kinect', 'realsense']:
        dest_dir = os.path.join(graspnet_root, 'scenes', 'scene_%04d' % sceneId, camera, 'rect')
        src_dir = os.path.join(rect_labels_root, 'scene_%04d' % sceneId, camera)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        for annId in range(256):
            src_path = os.path.join(src_dir,'%04d.npy' % annId)
            assert os.path.exists(src_path)
            os.system('cp {} {}'.format(src_path, dest_dir))
