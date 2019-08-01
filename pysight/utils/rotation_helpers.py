"""
Rotation Helpers
Collection of geometric, trigonometric, and black mathemagical C.V. functions for angle representation.
"""
from math import sin, asin, atan2, cos, sqrt
from cv2 import Rodrigues
from numpy import empty


def Euler2RotationMatrix(eulerAngles):
	"""
	Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign.
	Args:
		eulerAngles (Vec3d):
	Returns:
		rotation_matrix (Mat):
	"""
	# Doubles
	s1 = sin(eulerAngles[0])
	s2 = sin(eulerAngles[1])
	s3 = sin(eulerAngles[2])

	c1 = cos(eulerAngles[0])
	c2 = cos(eulerAngles[1])
	c3 = cos(eulerAngles[2])

	#rotation_matrix = [[None for j in range(3)] for i in range(3)]
	rotation_matrix = empty((3, 3))

	rotation_matrix[0][0] =  c2 * c3
	rotation_matrix[0][1] = -c2 * s3
	rotation_matrix[0][2] =  s2
	rotation_matrix[1][0] =  c1 * s3 + c3 * s1 * s2
	rotation_matrix[1][1] =  c1 * c3 - s1 * s2 * s3
	rotation_matrix[1][2] = -c2 * s1
	rotation_matrix[2][0] =  s1 * s3 - c1 * c3 * s2
	rotation_matrix[2][1] =  c3 * s1 + c1 * s2 * s3
	rotation_matrix[2][2] =  c1 * c2

	return rotation_matrix


def RotationMatrix2Euler(rotation_matrix):
	"""
	Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	Args:
		rotation_matrix (Matx33d)
	Returns:
		eulerAngles (Vec3d):
	"""

	q0 = sqrt(1 + rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2]) / 2.0
	q1 = (rotation_matrix[2][1] - rotation_matrix[1][2]) / (4.0 * q0)
	q2 = (rotation_matrix[0][2] - rotation_matrix[2][0]) / (4.0 * q0)
	q3 = (rotation_matrix[1][0] - rotation_matrix[0][1]) / (4.0 * q0)

	t1 		= 2.0 * (q0*q2 + q1*q3)

	yaw 	= asin(2.0 * (q0*q2 + q1*q3))
	pitch 	= atan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
	roll 	= atan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

	return [pitch, yaw, roll]


def Euler2AxisAngle(euler):
	"""
	Args:
		euler (Vec3d):
	Returns:
		axis_angle (float):
	"""
	rot_matrix = Euler2RotationMatrix(euler)
	axis_angle = Rodrigues(rot_matrix)
	return rot_matrix


def AxisAngle2Euler(axis_angle):
	"""Convenience function to return the rotation matrix of a given axis angle."""
	rot_matrix = Rodrigues(axis_angle)
	return rot_matrix


def AxisAngle2RotationMatrix(axis_angle):
	"""
	Args:
		axis_angle (Vec3d):
	Returns:
		rot_matrix (Mat33d):
	"""
	rot_matrix = Rodrigues(axis_angle)
	return rot_matrix


def RotationMatrix2AxisAngle(rotation_matrix):
	"""
	Args:
		rotation_matrix (Mat33d):
	Returns:
		axis_angle (Vec3d):
	"""
	axis_angle = Rodrigues(rotation_matrix)
	return axis_angle


"""def Project(mesh, fx, fy, cx, cy):
	"
	Args:
		mesh (Mat[double]):
		fx (double):
		fy (double):
		cx (double):
		cy (double):
	Returns:
		dest (Mat[double]):
	"
	# Number of rows
	num_points = len(mesh)

	# Instantiate output
	dest = [[0.0 for j in range(2)] for i in range(num_points)]

	for i in range(num_points):
		for j in range(2):
			# Get points
			X *= mesh[i][j]
			Y *= mesh[i][j]
			Z *= mesh[i][j]

			# If depth is 0, the projection is different.
			if Z != 0:
				x = ((X * fx / Z) + cx)
				y = ((Y * fy / Z) + cy)
			else:
				x = X
				y = Y
"""
