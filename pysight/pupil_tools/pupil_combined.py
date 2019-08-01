"""
PySight - Pupil Tools
"""

import cv2 as cv
import numpy as np

from .pupil_gradients import PupilGradients
from .pupil_isophotes import PupilIsophotes


class PupilTools(object):
    def __init__(self, w_grads=0.7, w_iso=0.3):
        self.w_grads         = w_grads
        self.w_iso           = w_iso
        self.pupil_gradients = PupilGradients()
        self.pupil_isophotes = PupilIsophotes()

    def __preprocess_eye_region(self, frame):
        """Helper function to blur and standard scale an eye image."""
        frame_grey  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur  = cv.GaussianBlur(frame_grey, (3, 3), 0, 0)
        frame_equal = cv.equalizeHist(frame_blur)
        return frame_equal

    def find_pupil(self, eye_img_bgr, fast_width_grads=50, fast_width_iso=80):
        """Find the (X, Y) center of a pupil from a given eye image."""

        # Preprocess
        eye_img_r = self.__preprocess_eye_region(eye_img_bgr)
        # eye_img_r = cv.cvtColor(eye_img_bgr, cv.COLOR_BGR2GRAY)

        # Resize Image for faster gradients computation
        fast_size_grads = (int((fast_width_grads / eye_img_bgr.shape[0]) * eye_img_bgr.shape[1]), int(fast_width_grads))
        fast_img_grads  = cv.resize(eye_img_r, fast_size_grads)

        # Resize Image for faster isophotes computation
        fast_size_iso = (int(fast_width_iso), int((fast_width_iso / eye_img_r.shape[1]) * eye_img_r.shape[0]))
        fast_img_iso = cv.resize(eye_img_r, fast_size_iso)

        # Create Mapping of gradients for each methodology
        c_map_grads = self.pupil_gradients.get_center_map(fast_img_grads)
        c_map_iso   = self.pupil_isophotes.get_center_map(fast_img_iso)

        # Normalize the mappings from each pupil finding method
        c_map_norm_grads = cv.normalize(c_map_grads, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        c_map_big_grads  = cv.resize(c_map_norm_grads, (eye_img_bgr.shape[1], eye_img_bgr.shape[0])).astype(np.uint8)

        c_map_norm_iso = cv.normalize(c_map_iso, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dst=None)
        c_map_big_iso  = cv.resize(c_map_norm_iso, (eye_img_bgr.shape[1], eye_img_bgr.shape[0])).astype(np.uint8)

        # Combine the methods to get a better estimate of pupil center
        joint_c_map = cv.addWeighted(c_map_big_grads, self.w_grads, c_map_big_iso, self.w_iso, 1.0)

        # Find the most likely candidate indices
        max_val_index = np.argmax(joint_c_map)
        pupil_y0, pupil_x0 = max_val_index // joint_c_map.shape[1], max_val_index % joint_c_map.shape[1]

        return pupil_x0, pupil_y0


    def get_pupil_coordinates(self, src, eye_rect):
        """Function to find the (x, y) center coordinates of a pupil in a
        eye ROI (Region of Interest) image.
        - src: face image encoded as NumPy array
        - eye_rect: rect denoting [x, y, w, h] of eye ROI.
        """
        # Tune these offsets to skip eyebrows and reduce window size
        x_off = eye_rect[0] #int(0.15f *)
        y_off = eye_rect[1] #int(0.40f *)
        box_w = eye_rect[2] #int(0.15f *)
        box_h = eye_rect[3] #int(0.40f *)

        pass
