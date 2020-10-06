from graspnetAPI import GraspNet
g = GraspNet('/home/gmh/graspnet', camera = 'realsense', split = 'all')
g.showSceneGrasp(5,'kinect',0,format = '6d')