__author__ = 'mhgou'

import numpy as np
import open3d as o3d
import copy
import cv2

from .utils.utils import plot_gripper_pro_max, batch_rgbdxyz_2_rgbxy_depth, get_batch_key_points, batch_key_points_2_tuple, framexy_depth_2_xyz, batch_framexy_depth_2_xyz, center_depth, key_point_2_rotation, batch_center_depth, batch_framexy_depth_2_xyz, batch_key_point_2_rotation

GRASP_ARRAY_LEN = 17
RECT_GRASP_ARRAY_LEN = 7
EPS = 1e-8

class Grasp():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id

        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        - the length of the numpy array is 17.
        '''
        if len(args) == 0:
            self.grasp_array = np.array([0, 0.02, 0.02, 0.02, 1, 0, 0, 0, 1 ,0 , 0, 0, 1, 0, 0, 0, -1], dtype = np.float64)
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == 7:
            score, width, height, depth, rotation_matrix, translation, object_id = args
            self.grasp_array = np.concatenate([np.array((score, width, height, depth)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)
        else:
            raise ValueError('only 1 or 7 arguments are accepted')
    
    def __repr__(self):
        return 'Grasp: score:{}, width:{}, height:{}, depth:{}, translation:{}\nrotation:\n{}\nobject id:{}'.format(self.score, self.width, self.height, self.depth, self.translation, self.rotation_matrix, self.object_id)

    @property
    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return float(self.grasp_array[0])

    @score.setter
    def score(self, score):
        '''
        **input:**

        - float of the score.
        '''
        self.grasp_array[0] = score

    @property
    def width(self):
        '''
        **Output:**

        - float of the width.
        '''
        return float(self.grasp_array[1])
    
    @width.setter
    def width(self, width):
        '''
        **input:**

        - float of the width.
        '''
        self.grasp_array[1] = width

    @property
    def height(self):
        '''
        **Output:**

        - float of the height.
        '''
        return float(self.grasp_array[2])

    @height.setter
    def height(self, height):
        '''
        **input:**

        - float of the height.
        '''
        self.grasp_array[2] = height
    
    @property
    def depth(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return float(self.grasp_array[3])

    @depth.setter
    def depth(self, depth):
        '''
        **input:**

        - float of the depth.
        '''
        self.grasp_array[3] = depth

    @property
    def rotation_matrix(self):
        '''
        **Output:**

        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.grasp_array[4:13].reshape((3,3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of matrix

        - len(args) == 9: float of matrix
        '''
        if len(args) == 1:
            self.grasp_array[4:13] = np.array(args[0],dtype = np.float64).reshape(9)
        elif len(args) == 9:
            self.grasp_array[4:13] = np.array(args,dtype = np.float64)

    @property
    def translation(self):
        '''
        **Output:**

        - np.array of shape (3,) of the translation.
        '''
        return self.grasp_array[13:16]

    @translation.setter
    def translation(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y, z

        - len(args) == 3: float of x, y, z
        '''
        if len(args) == 1:
            self.grasp_array[13:16] = np.array(args[0],dtype = np.float64)
        elif len(args) == 3:
            self.grasp_array[13:16] = np.array(args,dtype = np.float64)

    @property
    def object_id(self):
        '''
        **Output:**

        - int of the object id that this grasp grasps
        '''
        return int(self.grasp_array[16])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **Input:**

        - int of the object_id.
        '''
        self.grasp_array[16] = object_id

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)
        
        **Output:**

        - Grasp instance after transformation, the original Grasp will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translation = np.dot(rotation, self.translation.reshape((3,1))).reshape(-1) + translation
        self.rotation_matrix = np.dot(rotation, self.rotation_matrix)
        return self

    def to_open3d_geometry(self):
        '''
        **Ouput:**

        - list of open3d.geometry.Geometry of the gripper.
        '''
        return plot_gripper_pro_max(self.translation, self.rotation_matrix, self.width, self.depth, score = self.score)

class GraspGroup():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of grasp group array (3) str of the npy file.
        '''
        if len(args) == 0:
            self.grasp_group_array = np.zeros((0, GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __len__(self):
        '''
        **Output:**

        - int of the length.
        '''
        return len(self.grasp_group_array)

    def __repr__(self):
        repr = '----------\nGrasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array in self.grasp_group_array:
                repr += Grasp(grasp_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += Grasp(self.grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += Grasp(self.grasp_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**

        - index: int, slice, list or np.ndarray.

        **Output:**

        - if index is int, return Grasp instance.

        - if index is slice, np.ndarray or list, return GraspGroup instance.
        '''
        if type(index) == int:
            return Grasp(self.grasp_group_array[index])
        elif type(index) == slice:
            graspgroup = GraspGroup()
            graspgroup.grasp_group_array = copy.deepcopy(self.grasp_group_array[index])
            return graspgroup
        elif type(index) == np.ndarray:
            return GraspGroup(self.grasp_group_array[index])
        elif type(index) == list:
            return GraspGroup(self.grasp_group_array[index])
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for GraspGroup'.format(type(index)))

    @property
    def scores(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the scores.
        '''
        return self.grasp_group_array[:,0]
    
    @scores.setter
    def scores(self, scores):
        '''
        **Input:**

        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert scores.size == len(self)
        self.grasp_group_array[:,0] = copy.deepcopy(scores)

    @property
    def widths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the widths.
        '''
        return self.grasp_group_array[:,1]
    
    @widths.setter
    def widths(self, widths):
        '''
        **Input:**

        - widths: numpy array of shape (-1, ) of the widths.
        '''
        assert widths.size == len(self)
        self.grasp_group_array[:,1] = copy.deepcopy(widths)

    @property
    def heights(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the heights.
        '''
        return self.grasp_group_array[:,2]

    @heights.setter
    def heights(self, heights):
        '''
        **Input:**

        - heights: numpy array of shape (-1, ) of the heights.
        '''
        assert heights.size == len(self)
        self.grasp_group_array[:,2] = copy.deepcopy(heights)

    @property
    def depths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the depths.
        '''
        return self.grasp_group_array[:,3]

    @depths.setter
    def depths(self, depths):
        '''
        **Input:**

        - depths: numpy array of shape (-1, ) of the depths.
        '''
        assert depths.size == len(self)
        self.grasp_group_array[:,3] = copy.deepcopy(depths)

    @property
    def rotation_matrices(self):
        '''
        **Output:**

        - np.array of shape (-1, 3, 3) of the rotation matrices.
        '''
        return self.grasp_group_array[:, 4:13].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        '''
        **Input:**

        - rotation_matrices: numpy array of shape (-1, 3, 3) of the rotation_matrices.
        '''
        assert rotation_matrices.shape == (len(self), 3, 3)
        self.grasp_group_array[:,4:13] = copy.deepcopy(rotation_matrices.reshape((-1, 9)))       

    @property
    def translations(self):
        '''
        **Output:**

        - np.array of shape (-1, 3) of the translations.
        '''
        return self.grasp_group_array[:, 13:16]

    @translations.setter
    def translations(self, translations):
        '''
        **Input:**

        - translations: numpy array of shape (-1, 3) of the translations.
        '''
        assert translations.shape == (len(self), 3)
        self.grasp_group_array[:,13:16] = copy.deepcopy(translations)

    @property
    def object_ids(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the object ids.
        '''
        return self.grasp_group_array[:,16].astype(np.int32)

    @object_ids.setter
    def object_ids(self, object_ids):
        '''
        **Input:**

        - object_ids: numpy array of shape (-1, ) of the object_ids.
        '''
        assert object_ids.size == len(self)
        self.grasp_group_array[:,16] = copy.deepcopy(object_ids)

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)
        
        **Output:**

        - GraspGroup instance after transformation, the original GraspGroup will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translations = np.dot(rotation, self.translations.reshape((3,-1))).reshape((-1,3)) + translation # (-1, 3)
        self.rotation_matrices = np.matmul(rotation, self.rotation_matrices).reshape((-1, 3, 3)) # (-1, 9)
        return self

    def add(self, element):
        '''
        **Input:**

        - element: Grasp instance or GraspGroup instance.
        '''
        if isinstance(element, Grasp):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_array.reshape((-1, GRASP_ARRAY_LEN))))
        elif isinstance(element, GraspGroup):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_group_array))
        else:
            raise TypeError('Unknown type:{}'.format(element))
        return self

    def remove(self, index):
        '''
        **Input:**

        - index: list of the index of grasp
        '''
        self.grasp_group_array = np.delete(self.grasp_group_array, index, axis = 0)
        return self

    def from_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        self.grasp_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        np.save(npy_file_path, self.grasp_group_array)

    def to_open3d_geometry_list(self):
        '''
        **Output:**

        - list of open3d.geometry.Geometry of the grippers.
        '''
        geometry = []
        for i in range(len(self.grasp_group_array)):
            g = Grasp(self.grasp_group_array[i])
            geometry.append(g.to_open3d_geometry())
        return geometry
    
    def sort_by_score(self, reverse = False):
        '''
        **Input:**

        - reverse: bool of order, if False, from high to low, if True, from low to high.

        **Output:**

        - no output but sort the grasp group.
        '''
        score = self.grasp_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.grasp_group_array = self.grasp_group_array[index]
        return self

    def random_sample(self, numGrasp = 20):
        '''
        **Input:**

        - numGrasp: int of the number of sampled grasps.

        **Output:**

        - GraspGroup instance of sample grasps.
        '''
        if numGrasp > self.__len__():
            raise ValueError('Number of sampled grasp should be no more than the total number of grasps in the group')
        shuffled_grasp_group_array = copy.deepcopy(self.grasp_group_array)
        np.random.shuffle(shuffled_grasp_group_array)
        shuffled_grasp_group = GraspGroup()
        shuffled_grasp_group.grasp_group_array = copy.deepcopy(shuffled_grasp_group_array[:numGrasp])
        return shuffled_grasp_group

    def to_rect_grasp_group(self, camera):
        '''
        **Input:**

        - camera: string of type of camera, 'realsense' or 'kinect'.

        **Output:**
        
        - RectGraspGroup instance or None.
        '''
        tranlations = self.translations
        rotations = self.rotation_matrices
        depths = self.depths
        scores = self.scores
        widths = self.widths
        object_ids = self.object_ids

        mask = (rotations[:, 2, 0] > 0.99)
        tranlations = tranlations[mask]
        depths = depths[mask]
        widths = widths[mask]
        scores = scores[mask]
        rotations = rotations[mask]
        object_ids = object_ids[mask]
        
        if tranlations.shape[0] == 0:
            return None

        k_points = get_batch_key_points(tranlations, rotations, widths)
        k_points = k_points.reshape([-1, 3])
        k_points = k_points.reshape([-1, 4, 3])
        rect_grasp_group_array = batch_key_points_2_tuple(k_points, scores, object_ids, camera)
        rect_grasp_group = RectGraspGroup()
        rect_grasp_group.rect_grasp_group_array = rect_grasp_group_array
        return rect_grasp_group

    def nms(self, translation_thresh = 0.03, rotation_thresh = 30.0 / 180.0 * np.pi):
        '''
        **Input:**

        - translation_thresh: float of the translation threshold.

        - rotation_thresh: float of the rotation threshold.

        **Output:**

        - GraspGroup instance after nms.
        '''
        from grasp_nms import nms_grasp
        return GraspGroup(nms_grasp(self.grasp_group_array, translation_thresh, rotation_thresh))

class RectGrasp():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the center_x, center_y, open_x, open_y, height, score, object_id

        - the format of numpy array is [center_x, center_y, open_x, open_y, height, score, object_id]

        - the length of the numpy array is 7.
        '''
        if len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.rect_grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == RECT_GRASP_ARRAY_LEN:
            self.rect_grasp_array = np.array(args).astype(np.float64)
        else:
            raise ValueError('only one or six arguments are accepted')
    
    def __repr__(self):
        return 'Rectangle Grasp: score:{}, height:{}, open point:{}, center point:{}, object id:{}'.format(self.score, self.height, self.open_point, self.center_point, self.object_id)

    @property
    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return self.rect_grasp_array[5]
    
    @score.setter
    def score(self, score):
        '''
        **input:**

        - float of the score.
        '''
        self.rect_grasp_array[5] = score

    @property
    def height(self):
        '''
        **Output:**

        - float of the height.
        '''
        return self.rect_grasp_array[4]

    @height.setter
    def height(self, height):
        '''
        **input:**

        - float of the height.
        '''
        self.rect_grasp_array[4] = height

    @property
    def open_point(self):
        '''
        **Output:**

        - tuple of x,y of the open point.
        '''
        return (self.rect_grasp_array[2], self.rect_grasp_array[3])

    @open_point.setter
    def open_point(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y

        - len(args) == 2: float of x, y
        '''
        if len(args) == 1:
            self.rect_grasp_array[2:4] = np.array(args[0],dtype = np.float64)
        elif len(args) == 2:
            self.rect_grasp_array[2:4] = np.array(args,dtype = np.float64)

    @property
    def center_point(self):
        '''
        **Output:**

        - tuple of x,y of the center point.
        '''
        return (self.rect_grasp_array[0], self.rect_grasp_array[1])

    @center_point.setter
    def center_point(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y

        - len(args) == 2: float of x, y
        '''
        if len(args) == 1:
            self.rect_grasp_array[0:2] = np.array(args[0],dtype = np.float64)
        elif len(args) == 2:
            self.rect_grasp_array[0:2] = np.array(args,dtype = np.float64)

    @property
    def object_id(self):
        '''
        **Output:**

        - int of the object id that this grasp grasps
        '''
        return int(self.rect_grasp_array[6])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **input:**

        - float of the object_id.
        '''
        self.rect_grasp_array[6] = object_id

    def to_opencv_image(self, opencv_rgb):
        '''
        **input:**
        
        - opencv_rgb: numpy array of opencv BGR format.

        **Output:**

        - numpy array of opencv RGB format that shows the rectangle grasp.
        '''
        center_x, center_y, open_x, open_y, height, score, object_id = self.rect_grasp_array
        center = np.array([center_x, center_y])
        left = np.array([open_x, open_y])
        axis = left - center
        normal = np.array([-axis[1], axis[0]])
        normal = normal / np.linalg.norm(normal) * height / 2
        p1 = center + normal + axis
        p2 = center + normal - axis
        p3 = center - normal - axis
        p4 = center - normal + axis
        cv2.line(opencv_rgb, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 1, 8)
        cv2.line(opencv_rgb, (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])), (255,0,0), 3, 8)
        cv2.line(opencv_rgb, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 1, 8)
        cv2.line(opencv_rgb, (int(p4[0]),int(p4[1])), (int(p1[0]),int(p1[1])), (255,0,0), 3, 8)
        return opencv_rgb

    def get_key_points(self):
        '''
        **Output:**

        - center, open_point, upper_point, each of them is a numpy array of shape (2,)
        '''
        open_point = np.array(self.open_point)
        center = np.array(self.center_point)
        height = self.height
        open_point_vector = open_point - center
        unit_open_point_vector = open_point_vector / np.linalg.norm(open_point_vector)
        counter_clock_wise_rotation_matrix = np.array([[0,-1], [1, 0]])
        upper_point = np.dot(counter_clock_wise_rotation_matrix, unit_open_point_vector) * height / 2 + center
        return center, open_point, upper_point

    def to_grasp(self, camera, depths, depth_method = center_depth):
        '''
        **Input:**

        - camera: string of type of camera, 'kinect' or 'realsense'.

        - depths: numpy array of the depths image.

        - depth_method: function of calculating the depth.

        **Output:**

        - grasp: Grasp instance of None if the depth is not valid.
        '''
        center, open_point, upper_point = self.get_key_points()
        depth_2d = depth_method(depths, center, open_point, upper_point) / 1000.0
        # print('depth 2d:{}'.format(depth_2d))
        if abs(depth_2d) < EPS:
            return None
        center_xyz = np.array(framexy_depth_2_xyz(center[0], center[1], depth_2d, camera))
        open_point_xyz = np.array(framexy_depth_2_xyz(open_point[0], open_point[1], depth_2d, camera))
        upper_point_xyz = np.array(framexy_depth_2_xyz(upper_point[0], upper_point[1], depth_2d, camera))
        depth = 0.02
        height = np.linalg.norm(upper_point_xyz - center_xyz) * 2
        width = np.linalg.norm(open_point_xyz - center_xyz) * 2 
        score = self.score
        object_id = self.object_id
        translation = center_xyz
        rotation = key_point_2_rotation(center_xyz, open_point_xyz, upper_point_xyz)
        # to avoid bug some time
        if height < EPS:
            return None
        return Grasp(score, width, height, depth, rotation, translation, object_id)

class RectGraspGroup():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of rect_grasp_group_array (3) str of the numpy file.
        '''
        if len(args) == 0:
            self.rect_grasp_group_array = np.zeros((0, RECT_GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.rect_grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.rect_grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __len__(self):
        '''
        **Output:**

        - int of the length.
        '''
        return len(self.rect_grasp_group_array)
    
    def __repr__(self):
        repr = '----------\nRectangle Grasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 10:
            for rect_grasp_array in self.rect_grasp_group_array:
                repr += RectGrasp(rect_grasp_array).__repr__() + '\n'
        else:
            for i in range(5):
                repr += RectGrasp(self.rect_grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(5):
                repr += RectGrasp(self.rect_grasp_group_array[-(5-i)]).__repr__() + '\n'
        return repr + '----------'
            
    def __getitem__(self, index):
        '''
        **Input:**

        - index: int, slice, list or np.ndarray.

        **Output:**

        - if index is int, return Grasp instance.

        - if index is slice, np.ndarray or list, return RectGraspGroup instance.
        '''
        if isinstance(index, int):
            return RectGrasp(self.rect_grasp_group_array[index])
        elif isinstance(index, slice) or isinstance(index, list) or isinstance(index, np.ndarray):
            rectgraspgroup = RectGraspGroup()
            rectgraspgroup.rect_grasp_group_array = copy.deepcopy(self.rect_grasp_group_array[index])
            return rectgraspgroup
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for RectGraspGroup'.format(type(index)))

    def add(self, rect_grasp):
        '''
        **Input:**

        - rect_grasp: RectGrasp instance
        '''
        self.rect_grasp_group_array = np.concatenate((self.rect_grasp_group_array, rect_grasp.rect_grasp_array.reshape((-1, RECT_GRASP_ARRAY_LEN))))
        return self

    @property
    def scores(self):
        '''
        **Output:**

        - numpy array of the scores.
        '''
        return self.rect_grasp_group_array[:, 5]

    @scores.setter
    def scores(self, scores):
        '''
        **Input:**

        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert scores.size == len(self)
        self.rect_grasp_group_array[:, 5] = copy.deepcopy(scores)

    @property
    def heights(self):
        '''
        **Output:**

        - numpy array of the heights.
        '''
        return self.rect_grasp_group_array[:, 4]

    @heights.setter
    def heights(self, heights):
        '''
        **Input:**

        - heights: numpy array of shape (-1, ) of the heights.
        '''
        assert heights.size == len(self)
        self.rect_grasp_group_array[:, 4] = copy.deepcopy(heights)

    @property
    def open_points(self):
        '''
        **Output:**

        - numpy array the open points of shape (-1, 2).
        '''
        return self.rect_grasp_group_array[:, 2:4]

    @open_points.setter
    def open_points(self, open_points):
        '''
        **Input:**

        - open_points: numpy array of shape (-1, 2) of the open_points.
        '''
        assert open_points.shape == (len(self), 2)
        self.rect_grasp_group_array[:, 2:4] = copy.deepcopy(open_points)

    @property
    def center_points(self):
        '''
        **Output:**

        - numpy array the center points of shape (-1, 2).
        '''
        return self.rect_grasp_group_array[:, 0:2]

    @center_points.setter
    def center_points(self, center_points):
        '''
        **Input:**

        - center_points: numpy array of shape (-1, 2) of the center_points.
        '''
        assert center_points.shape == (len(self), 2)
        self.rect_grasp_group_array[:, 0:2] = copy.deepcopy(center_points)

    @property
    def object_ids(self):
        '''
        **Output:**

        - numpy array of the object ids that this grasp grasps.
        '''
        return np.round(self.rect_grasp_group_array[:, 6]).astype(np.int32)

    @object_ids.setter
    def object_ids(self, object_ids):
        '''
        **Input:**

        - heiobject_idsghts: numpy array of shape (-1, ) of the object_ids.
        '''
        assert object_ids.size == len(self)
        self.rect_grasp_group_array[:, 6] = copy.deepcopy(object_ids)

    def remove(self, index):
        '''
        **Input:**

        - index: list of the index of rect_grasp
        '''
        self.rect_grasp_group_array = np.delete(self.rect_grasp_group_array, index, axis = 0)

    def from_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        self.rect_grasp_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        np.save(npy_file_path, self.rect_grasp_group_array)

    def to_opencv_image(self, opencv_rgb, numGrasp = 0):
        '''
        **input:**
        
        - opencv_rgb: numpy array of opencv BGR format.

        - numGrasp: int of the number of grasp, 0 for all.

        **Output:**

        - numpy array of opencv RGB format that shows the rectangle grasps.
        '''
        img = copy.deepcopy(opencv_rgb)
        if numGrasp == 0:
            numGrasp = self.__len__()
        shuffled_rect_grasp_group_array = copy.deepcopy(self.rect_grasp_group_array)
        np.random.shuffle(shuffled_rect_grasp_group_array)
        for rect_grasp_array in shuffled_rect_grasp_group_array[:numGrasp]:
            center_x, center_y, open_x, open_y, height, score, object_id = rect_grasp_array
            center = np.array([center_x, center_y])
            left = np.array([open_x, open_y])
            axis = left - center
            normal = np.array([-axis[1], axis[0]])
            normal = normal / np.linalg.norm(normal) * height / 2
            p1 = center + normal + axis
            p2 = center + normal - axis
            p3 = center - normal - axis
            p4 = center - normal + axis
            cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 1, 8)
            cv2.line(img, (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])), (255,0,0), 3, 8)
            cv2.line(img, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 1, 8)
            cv2.line(img, (int(p4[0]),int(p4[1])), (int(p1[0]),int(p1[1])), (255,0,0), 3, 8)
        return img

    def batch_get_key_points(self):
        '''
        **Output:**

        - center, open_point, upper_point, each of them is a numpy array of shape (2,)
        '''
        open_points = self.open_points # (-1, 2)
        centers = self.center_points # (-1, 2)
        heights = (self.heights).reshape((-1, 1)) # (-1, )
        open_point_vector = open_points - centers
        norm_open_point_vector = np.linalg.norm(open_point_vector, axis = 1).reshape(-1, 1)
        unit_open_point_vector = open_point_vector / np.hstack((norm_open_point_vector, norm_open_point_vector)) # (-1, 2)
        counter_clock_wise_rotation_matrix = np.array([[0,-1], [1, 0]])
        upper_points = np.dot(counter_clock_wise_rotation_matrix, unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack([heights, heights]) / 2 + centers # (-1, 2)
        return centers, open_points, upper_points

    def to_grasp_group(self, camera, depths, depth_method = batch_center_depth):
        '''
        **Input:**

        - camera: string of type of camera, 'kinect' or 'realsense'.

        - depths: numpy array of the depths image.

        - depth_method: function of calculating the depth.

        **Output:**

        - grasp_group: GraspGroup instance or None.

        .. note:: The number may not be the same to the input as some depth may be invalid.
        '''
        centers, open_points, upper_points = self.batch_get_key_points()
        # print(f'centers:{centers}\nopen points:{open_points}\nupper points:{upper_points}')
        depths_2d = depth_method(depths, centers, open_points, upper_points) / 1000.0
        # print(f'depths_3d:{depths_2d}')
        valid_mask1 = np.abs(depths_2d) > EPS
        valid_mask2 = np.linalg.norm(centers - open_points, axis =1) > EPS
        valid_mask3 = np.linalg.norm(centers - upper_points, axis =1) > EPS
        valid_mask4 = np.linalg.norm(upper_points - open_points, axis =1) > EPS
        valid_mask = np.logical_and(
            np.logical_and(valid_mask1, valid_mask2),
            np.logical_and(valid_mask3, valid_mask4)
        )
        # print(f'valid_mask:{valid_mask}')
        centers = centers[valid_mask]
        open_points = open_points[valid_mask]
        upper_points = upper_points[valid_mask]
        # print(f'## After filtering\ncenters:{centers}\nopen points:{open_points}\nupper points:{upper_points}')
        depths_2d = depths_2d[valid_mask]
        valid_num = centers.shape[0]
        if valid_num == 0:
            return None
        centers_xyz = np.array(batch_framexy_depth_2_xyz(centers[:, 0], centers[:, 1], depths_2d, camera)).T
        open_points_xyz = np.array(batch_framexy_depth_2_xyz(open_points[:, 0], open_points[:, 1], depths_2d, camera)).T
        upper_points_xyz = np.array(batch_framexy_depth_2_xyz(upper_points[:, 0], upper_points[:, 1], depths_2d, camera)).T
        depths = 0.02 * np.ones((valid_num, 1))
        heights = (np.linalg.norm(upper_points_xyz - centers_xyz, axis = 1) * 2).reshape((-1, 1))
        widths = (np.linalg.norm(open_points_xyz - centers_xyz, axis = 1) * 2).reshape((-1, 1))
        scores = (self.scores)[valid_mask].reshape((-1, 1))
        object_ids = (self.object_ids)[valid_mask].reshape((-1, 1))
        translations = centers_xyz
        rotations = batch_key_point_2_rotation(centers_xyz, open_points_xyz, upper_points_xyz).reshape((-1, 9))
        grasp_group = GraspGroup()
        grasp_group.grasp_group_array = copy.deepcopy(np.hstack((scores, widths, heights, depths, rotations, translations, object_ids))).astype(np.float64)
        return grasp_group

    def sort_by_score(self, reverse = False):
        '''
        **Input:**

        - reverse: bool of order, if False, from high to low, if True, from low to high.

        **Output:**

        - no output but sort the grasp group.
        '''
        score = self.rect_grasp_group_array[:,5]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.rect_grasp_group_array = self.rect_grasp_group_array[index]
        return self

    def random_sample(self, numGrasp = 20):
        '''
        **Input:**

        - numGrasp: int of the number of sampled grasps.

        **Output:**

        - RectGraspGroup instance of sample grasps.
        '''
        if numGrasp > self.__len__():
            raise ValueError('Number of sampled grasp should be no more than the total number of grasps in the group')
        shuffled_rect_grasp_group_array = copy.deepcopy(self.rect_grasp_group_array)
        np.random.shuffle(shuffled_rect_grasp_group_array)
        shuffled_rect_grasp_group = RectGraspGroup()
        shuffled_rect_grasp_group.rect_grasp_group_array = copy.deepcopy(shuffled_rect_grasp_group_array[:numGrasp])
        return shuffled_rect_grasp_group