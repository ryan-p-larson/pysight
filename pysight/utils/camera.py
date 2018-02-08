"""
PySight - Camera class
"""

# OpenCV resize docs mention INTER_AREA is best for shrinking images, linear is fater
from cv2 import VideoCapture, COLOR_BGR2GRAY, equalizeHist, cvtColor, resize, INTER_LINEAR, INTER_AREA

class Camera(object):
	"""Helper class that eases the setup/teardown of an OpenCV webcam."""
	def __init__(self, src=0):
		"""Instantiate a Webcam Video Stream from given src."""
		self.__src    = src
		self.__frames = []

	def capture(self):
		"""Function to start, capture, and shutdown our camera.
		Returns:
			frame (np.ndarray/None): returns image encoded as numpy array if 
			valid, else None
		"""
		
		webcam = VideoCapture(self.__src) 	# Fire up our webcam
		grabbed, frame = webcam.read()      # Grab an image as uint8 array.
		webcam.release()                    # Stop our webcam so we can use it again.
		
		return frame if grabbed else None

	def process(self, img, downsize=5):
		"""Function to process an raw RGB image into an B&W, equalized img.
		Args:
			img (np.array): Image in the form of a np.nd? array, typically from self.poll()
		Returns:
			img_equalized (np.ndarray): Processed image.
		"""

		# Down sample image so our algo has to do less work.
		# we need to keep in mind aspect ratio so the image doesn't look distorted.
		height, width = img.shape[:2]
		down_height, down_width = int(height / downsize), int(width/ downsize)
		dim = (down_width, down_height) if (downsize != 1) else (width, height)
 
		img_scaled		= resize(img, dim, interpolation=INTER_AREA)# Resize with our calc'd
		img_gray 		= cvtColor(img_scaled, COLOR_BGR2GRAY) 		# Convert to grayscale.
		img_equalized 	= equalizeHist(img_gray)					# Equalize the values
		return img_equalized