"""
PySight - Test Face Tools module.
Tests...
	find_faces
	find_eyes
	find_pupils
	find_shape
"""

import unittest
import pysight
from cv2 import imread

## FACE --------------------------------------------------------------------

class FaceToolsTestCase(unittest.TestCase):
	"""Tests for face_tools.py."""


	def setUp(self):
		"""Sets up a common FaceTools object to run our tests on."""

		# Paths to our classifers
		face_f = 'pysight/models/haarcascade_frontalface_alt.xml'
		eyes_f = 'pysight/models/haarcascade_eye.xml'
		dlib_f = 'pysight/models/shape_predictor_68_face_landmarks.dat'

		self.face_tools = pysight.FaceTools(face_f, eyes_f, dlib_f)


	def test_ft_models_instantiation(self):
		"""Tests if our Tool object can sucessfully load 
		and find our models."""

		# Assert that each OpenCV Caascade is loaded
		self.assertFalse(self.face_tools.face_clf.empty())
		self.assertFalse(self.face_tools.eyes_clf.empty())

		# Assert Dlib Shape Predictor is loaded
		# TO-DO
		#


	def test_ft_face_clf_good_single(self):
		"""Tests if our face bounding box classifier can detect a face
		in a known good photo."""

		# Path to an image with a clearly represented face.
		img_good_path = imread('tests/test_imgs/webcam-single.jpg')

		# Run the facial detection classifier.
		face_in_img = self.face_tools.find_faces(img_good_path)

		# Run test
		self.assertTrue(len(face_in_img) == 1)


	def test_ft_face_clf_good_multiple(self):
		"""Tests if our facial classifier can detect multiple faces 
		in a known good photo."""

		# Path to an image with a clearly represented face.
		img_good_path = imread('tests/test_imgs/webcam-multiple.jpg')

		# Run the facial detection classifier.
		faces_in_img = self.face_tools.find_faces(img_good_path, multi=True)

		# Run test
		self.assertTrue(len(faces_in_img) > 1)


	def test_ft_face_clf_bad(self):
		"""Tests if our facial classifier will gracefully process 
		an image with no faces."""

		# Path to an image with a clearly represented face.
		img_bad_path = imread('tests/test_imgs/webcam-bad-face.jpg')

		# Run the facial detection classifier.
		faces_in_img = self.face_tools.find_faces(img_bad_path)

		# Run test
		self.assertIsNotNone(faces_in_img)


	def test_ft_eyes_clf_good(self):
		"""Tests if our FaceTools object can detect eyes in a known good image."""
		img_path = imread('tests/test_imgs/webcam-single.jpg')
		img_processed = pysight.Camera().process(img_path)

		# TO-DO: Add face rect bound
		# face_in_img = self.face_tools.find_faces(img_processed)

		# Run eyes classifier.
		eyes_in_img = self.face_tools.find_eyes(img_processed)

		# Test!
		self.assertNotEqual(eyes_in_img, ())


	def test_ft_eyes_clf_bad(self):
		"""Tests if our FaceTools object can gracefully (not) detect eyes in 
		a known bad image."""
		img_path = imread('tests/test_imgs/webcam-bad-eyes.jpg')

		# Run eyes classifier.
		eyes_in_img = self.face_tools.find_eyes(img_path)

		# Test!
		self.assertEqual(eyes_in_img, ())


	def test_ft_pupils_good(self):
		"""Test to determine if out FaceTools object can detect pupils
		in a known good photo."""

		# Path to an image with a clearly represented eye and pupil.
		img_good = imread('tests/test_imgs/eye-single.jpg')
		
		# Image MUST be processed into gray for Hough Circles to work
		img_proc = pysight.Camera().process(img_good)

		# Find the pupils
		pupils = self.face_tools.find_pupils(img_proc)

		# Test!
		self.assertNotEqual(pupils, ())


	def test_ft_pupils_bad(self):
		"""Test to determine if our find_pupils method can gracefully process
		pupils from a known bad image."""
		
		pass


	def test_ft_dlib_clf_good(self):
		"""Test if our Dlib shape predictor can find 68 landmarks in 
		a known good facial picture."""

		# Path to an image with a clearly represented face.
		img_good_path = imread('tests/test_imgs/webcam-single.jpg')

		# First find facial bounding box
		face_rect = self.face_tools.find_faces(img_good_path)

		# Find landmarks on face
		landmarks = self.face_tools.find_landmarks(img_good_path, face_rect)

		# Test
		self.assertNotEqual(landmarks, [])


	def test_ft_dlib_clf_bad(self):
		"""Tests if our Dlib shape predictor can gracefully process landmarks
		from a known bad image."""
		
		# Path to an image with a clearly represented face.
		img_good_path = imread('tests/test_imgs/webcam-bad-eyes.jpg')

		# First find facial bounding box
		face_rect = self.face_tools.find_faces(img_good_path)

		# Find landmarks on face
		landmarks = self.face_tools.find_landmarks(img_good_path, face_rect)

		# Test
		self.assertEqual(landmarks, [])


if __name__ == '__main__':
	unittest.main()
