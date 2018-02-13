"""
PySight - Pupil Tools
"""

import cv2 as cv
import numpy as np

class PupilGradients(object):
    def __init__(self, fast_width=50.0, inv_intensity_weight_divisor=100):
        """Helper class.
            fast_width (int): standardized width of resized eye images
            inv_intensity_weight_divisor (int): Param defining normalization 
                of image darkness weighting."""
        self.fast_width = fast_width
        self.inv_intensity_weight_divisor = inv_intensity_weight_divisor


    def __test_possible_centers_formula(self, grad_x0, grad_y0, darkness_weight, 
                                        grad_x_val, grad_y_val, indicies_grid, shape):
        """Algorithm from "ACCURATE EYE CENTRE LOCALISATION BY MEANS OF GRADIENTS 
        - Fabian Timm and Erhardt Barth"""
        #
        dx = np.ones(shape) * grad_x0 - indicies_grid[1]
        dy = np.ones(shape) * grad_y0 - indicies_grid[0]
        magnitudes = cv.magnitude(dx + 0.0001, dy)          # 0.0001 is a hack to offset against division by 0
        
        # Normalize the vectors
        dx = dx / magnitudes
        dy = dy / magnitudes
        
        # Compute and filter the valid deltas
        diffs = (dx * grad_x_val + dy * grad_y_val) * darkness_weight
        diffs[diffs < 0] = 0   
        
        return diffs


    def get_center_map(self, eye_img_grey):
        """Function to find the pupil center candidates from an eye ROI image."""
        # Compute gradients along X, Y axes
        grad_x_img = cv.Sobel(eye_img_grey, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
        grad_y_img = cv.Sobel(eye_img_grey, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
        
        # Compute magnitudes of gradients, and the threshold based of mean + std dev
        magnitudes = np.sqrt(grad_x_img ** 2 + grad_y_img ** 2).astype(int)
        mag_thresh = int(np.std(magnitudes) / 2 + np.mean(magnitudes))
        
        # Normalize the gradients based on threshold
        grad_x_img = np.divide(grad_x_img, magnitudes + 1)
        grad_y_img = np.divide(grad_y_img, magnitudes + 1)
        
        # Filter out gradients below threshold
        grad_x_img[magnitudes < mag_thresh] = 0
        grad_y_img[magnitudes < mag_thresh] = 0   
        
        # Invert image for weighting
        #eye_img_inv = 255 - eye_img_grey
        #weighted_img = cv.GaussianBlur(eye_img_grey, (3, 3), 0, 0)
        eye_img_inv = cv.bitwise_not(eye_img_grey)
        
        # Normalize the weights
        darkness_weights = eye_img_inv / self.inv_intensity_weight_divisor
        
        # Create an matrix that will hold our pupil center positions
        accumulator = np.zeros(eye_img_grey.shape[:2], dtype=np.float32)
        indicies_grid = np.indices(accumulator.shape[:2])
        indicies_shape = indicies_grid.shape[1:3]
        
        for y in range(eye_img_grey.shape[0]):
            for x in range (eye_img_grey.shape[1]):
                if grad_x_img[y][x] == 0 and grad_y_img[y][x] == 0: continue
                accumulator += self.__test_possible_centers_formula(x, y,
                                                            darkness_weights[y][x],
                                                            grad_x_img[y][x], grad_y_img[y][x],
                                                            indicies_grid, indicies_shape)
        
        num_gradients = eye_img_grey.shape[0] * eye_img_grey.shape[1]
        return accumulator * (1 / num_gradients)    


    def find_pupil(self, eye_img_bgr):
        """ Estimates the centre of the pupil using image gradients.
        """


        #eye_img_r = cv.split(eye_img_bgr)[2]   # Extract red channel only
        eye_img_r = self.__preprocess_eye_region(eye_img_bgr.copy())
        
        # Scale to small image for faster computation
        scale = self.fast_width / eye_img_bgr.shape[0]
        small_size = (int((self.fast_width / eye_img_bgr.shape[0]) * eye_img_bgr.shape[1]), int(self.fast_width))
        eye_img_small = cv.resize(eye_img_r, small_size)
        
        center_map = self.get_center_map(eye_img_small)
        
        max_val_index = np.argmax(center_map)
        pupil_y0, pupil_x0 = max_val_index // center_map.shape[1], max_val_index % center_map.shape[1]
        
        # Scale back to original coordinates
        pupil_y0, pupil_x0 = int((pupil_y0 + 0.5) / scale), int((pupil_x0 + 0.5) / scale)
        return pupil_x0, pupil_y0
