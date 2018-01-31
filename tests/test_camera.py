"""
PySight - Test Camera Module
Tests...
	camera opening
	camera processing
		size attr
		empty img
		None image
"""

import unittest
import pysight

class CameraTestCase(unittest.TestCase):
	"""Tests for camera.py"""

	def setUp(self):
		"""Function that creates a Camera object for oue tests."""
		self.webcam = pysight.Camera()


	def test_does_camera_open_good(self):
		"""Tests if Camera() can open a good webcam. This will open the default 
		webcam: /dev/video0. 
		REQUIRES A WEBCAM TO BE CONNECTED TO PASS!
		"""
		self.assertTrue(self.webcam.isOpened())

	
	def test_does_camera_capture(self):
		"""Tests if our OpenCV camera class can take a photo."""

		# Grab a single frame from our webcam
		frame = self.webcam.capture()
		self.assertTrue(frame != None)


	def test_can_camera_capture_mult(self):
		"""Tests if our camera can take multiple screen grabs.
		Because the camera will be opened/closed multiple times throughout a
		session, we need to make sure it will take multiple photos and still
		be responsive. Essentially this tests that our camera closes well."""

		frames = []
		for i in range(5):
			frames.append(self.webcam.capture())

		# TO-DO: make sure each frame is valid
		self.assertTrue(len(frames) == 5)


	def test_does_camera_process_img(self):
		"""Tests if our camera can correctly process an img."""
		pass


	def test_does_camera_process_noimg(self):
		"""Tests if our camera can correctly process an empty img
		without blowing up with errors."""
		pass


	def test_does_camera_process_size(self):
		"""Tests if our camera can correctly resize an img with param.
		Must retain aspect ratio!"""
		# 1/5
		# 1/3
		# 1/1
		pass


if __name__ == '__main__':
	unittest.main()