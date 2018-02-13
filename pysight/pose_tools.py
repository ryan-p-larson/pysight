"""
PySight - Pose Tools
"""

import cv2 as cv
import dlib


class PoseTools(object):
    def __init__(self):
        self.dlib_clf = None
        


    def cv_point_2_dlib(self, point):
        """Helper function to convert an OpenCV point to Dlib format (dlib.point)."""
        return dlib.point(point[0], point[1])


    def dlib_point_2_cv(self, point):
        """Helper function to convert a Dlib point to OpenCV format (tuple)."""
        return (point.x, point.y)


    def dlib_shape_2_cv(self, shape):
        """Helper function that converts an active appearance model to opencv."""
        return [self.dlib_point_2_cv(p) for p in shape.parts()]




    # def get_largest_face
    #   passed an array of bounding boxes from face_tools + convert
    # def get_left_eye
    # def get_right_eye
    # 