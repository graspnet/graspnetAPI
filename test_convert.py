from graspnetAPI import GraspNet
import cv2
import open3d as o3d

camera = 'kinect'
sceneId = 5
annId = 3

g = GraspNet('/home/gmh/graspnet', camera = camera, split = 'all')

rect_grasp_group = g.loadGrasp(sceneId = sceneId, camera = camera, annId = annId, fric_coef_thresh = 0.2, format = 'rect')
rect_grasp = rect_grasp_group.random_sample(1)[0]
bgr = g.loadBGR(sceneId = sceneId, camera = camera, annId = annId)
img = rect_grasp.to_opencv_image(bgr)
depth = g.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
cv2.imshow('grasp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

grasp = rect_grasp.to_grasp(camera, depth)
geometry = []
geometry.append(g.loadScenePointCloud(sceneId, camera, annId))
geometry.append(grasp.to_open3d_geometry())
o3d.visualization.draw_geometries(geometry)