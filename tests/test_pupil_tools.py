"""
PySight - Test Pupil Tools module.
Tests...
	Find center - gradients
    Find center - isophotes
    Find center - combined

        Control image
	
"""

import unittest
from cv2 import imread
import pysight

from pysight.pupil_tools.pupil_combined import PupilTools
from pysight.pupil_tools.pupil_gradients import PupilGradients
from pysight.pupil_tools.pupil_isophotes import PupilIsophotes


## FACE --------------------------------------------------------------------

class FaceToolsTestCase(unittest.TestCase):
	"""Tests for face_tools.py."""


	def setUp(self):
		"""Sets up a common PupilTools object to run our tests on."""

		# Paths to our different types of pupil detectors
		self.pupil_gradients = PupilGradients()
        self.pupil_isophotes = PupilIsophotes()
        self.pupil_combined  = PupilTools()

        # Add the locations of images to test
        #self.control_pupil_img =
        #self.good_pupil_img
        #self.bad_pupil_img


	


if __name__ == '__main__':
	unittest.main()
