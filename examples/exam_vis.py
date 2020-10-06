__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for loading grasp for a scene.
# change the graspnet_root path

####################################################################
graspnet_root = '/home/gmh/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

# sceneId = 1
from graspnetAPI import GraspNet

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# show object grasps
g.showObjGrasp(objIds = 0, show=True)

# show 6d poses
g.show6DPose(sceneIds=0, show=True)

# show scene rectangle grasps
# g.showSceneGrasp(sceneIds=0, format = 'rect', show = True, numGrasp = 20)

# show scene 6d grasps(You may need to wait several minutes)
g.showSceneGrasp(5,camera = 'kinect',annId = 0,format = '6d')