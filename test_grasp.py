from graspnetAPI import GraspNet
import cv2
camera = 'kinect'
g = GraspNet('/home/gmh/graspnet', camera = camera, split = 'all')
# grasp_group = g.loadGrasp(0, grasp_thresh=0.2)
# print(grasp_group)
# d = g.loadDepth(0, 'kinect', 0)

# from graspnetAPI import GraspGroup
# import open3d as o3d
# g = GraspGroup()
# g.from_npy('s0a0kinect6d.npy')
# gmt = g[:100].to_open3d_geometry_list()

# o3d.visualization.draw_geometries(gmt)
# print(g)

grasp_group = g.loadGrasp(sceneId = 5, camera = camera, annId = 3, grasp_thresh=0.2, format = '6d')
rect_grasp_group = grasp_group.to_rect_grasp_group(camera)
bgr = g.loadBGR(sceneId=5, camera = camera, annId = 3)
img = rect_grasp_group.to_opencv_image(bgr, num_grasp = 20)
cv2.imshow('bgr', img)
cv2.waitKey(0)
cv2.destroyAllWindows()