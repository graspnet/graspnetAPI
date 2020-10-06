from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
import numpy as np

def get_pose(pose):
	pos, quat = pose_4x4_to_pos_quat(pose)
	euler = np.array([quat2euler(quat)[0], quat2euler(quat)[1],quat2euler(quat)[2]])
	euler = euler * 180.0 / np.pi
	alpha, beta, gamma = euler[0], euler[1], euler[2]
	x, y, z = pos[0], pos[1], pos[2]
	return x,y,z, alpha, beta, gamma

def get_mat(x,y,z, alpha, beta, gamma):
	"""
	Calls get_mat() to get the 4x4 matrix
	"""
	try:
		euler = np.array([alpha, beta, gamma]) / 180.0 * np.pi
		quat = np.array(euler2quat(euler[0],euler[1],euler[2]))
		pose = pos_quat_to_pose_4x4(np.array([x,y,z]), quat)
		return pose
	except Exception as e:
		print(str(e))
		pass         

def pos_quat_to_pose_4x4(pos, quat):
	"""pose = pos_quat_to_pose_4x4(pos, quat)
	Convert pos and quat into pose, 4x4 format

	Args:
	    pos: length-3 position
	    quat: length-4 quaternion

	Returns:
	    pose: numpy array, 4x4
	"""
	pose = np.zeros([4, 4])
	mat = quat2mat(quat)
	pose[0:3, 0:3] = mat[:, :]
	pose[0:3, -1] = pos[:]
	pose[-1, -1] = 1
	return pose


def pose_4x4_to_pos_quat(pose):
	"""
	Convert pose, 4x4 format into pos and quat

	Args:
	    pose: numpy array, 4x4
	Returns:
		pos: length-3 position
	    quat: length-4 quaternion

	"""
	mat = pose[:3, :3]
	quat = mat2quat(mat)
	pos = np.zeros([3])
	pos[0] = pose[0, 3]
	pos[1] = pose[1, 3]
	pos[2] = pose[2, 3]
	return pos, quat