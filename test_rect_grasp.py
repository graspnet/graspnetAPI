import numpy as np
import cv2
from graspnetAPI.grasp import RectGraspGroup, RectGrasp
rgb = cv2.imread('/home/gmh/graspnet/scenes/scene_0000/kinect/rgb/0000.png')
rgg = RectGraspGroup()


# rgg.rect_grasp_group_array = np.zeros((866,7),dtype=np.float32)
# rgg.rect_grasp_group_array[:,:6] = np.load('/home/gmh/graspnet/scenes/scene_0000/kinect/rectangle_grasp/0000.npy')
# rgg.rect_grasp_group_array[:,6]  = -1 * np.ones((866), dtype=np.float32)
# # from_npy('/home/minghao/graspnet/scenes/scene_0000/kinect/rectangle_grasp/0000.npy')
# # rgg = RectGraspGroup(r)
# rgg.sort_by_score()
# rgg.save_npy('s0a0kinect.npy')


rgg.from_npy('s0a0kinect.npy')
print(rgg)

rgg0 = RectGraspGroup()
rgg0.add(rgg[0])
img = rgg.to_opencv_image(rgb, num_grasp=10)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()