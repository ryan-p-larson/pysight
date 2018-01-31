"""
PySight - Camera class
"""

# from cv2 import ...

class Camera(object):
	"""Helper class that eases the setup/teardown of an OpenCV webcam."""
	def __init__(self, src=0):
		"""Instantiate a Webcam Video Stream from given src."""
		self.__src    = src
		self.__frames = []

	def capture(self):
		"""Function to start, capture, and shutdown our camera."""
		
		webcam = cv2.VideoCapture(self.__src) # Fire up our webcam
		grabbed, frame = webcam.read()      # Grab an image as uint8 array.
		webcam.release()                    # Stop our webcam so we can use it again.
		
		return frame

	def process(self, img):
		"""Function to process an RGB image into an B&W, equalized img."""

		# resize image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 	# Convert to grayscale.
		equalized = cv2.equalizeHist(gray)				# Equalize the values
		return equalized