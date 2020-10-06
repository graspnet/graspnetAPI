from graspnetAPI import GraspNet

if __name__ == '__main__':

    ####################################################################
    graspnet_root = '/home/gmh/graspnet'  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    g = GraspNet(graspnet_root, 'kinect', 'train')
    if g.check_data_completeness():
        print('Check for kinect passed')


    g = GraspNet(graspnet_root, 'realsense', 'test_seen')
    if g.check_data_completeness():
        print('Check for kinect passed')