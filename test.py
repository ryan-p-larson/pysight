"""
PySight - Test suites
"""

# Make sure our package imports
import pysight


## CAMERA --------------------------------------------------------------------

# Set up a webcam
webcam = pysight.Camera()

# Take a photo
img = webcam.capture()

# Process photo
processed_img = webcam.process(img, 1)

print ("Camera sucessful!")


## FACE --------------------------------------------------------------------

# Paths to our classifers
face_f = 'pysight/models/haarcascade_frontalface_alt.xml'
eyes_f = 'pysight/models/haarcascade_eye.xml'
dlib_f = 'pysight/models/shape_predictor_68_face_landmarks.dat'

# Instantiate a face tools object
face_tools = pysight.Face(face_f, eyes_f, dlib_f)

# Check for faces
faces_in_img = face_tools.find_faces(processed_img)

# Check for eyes
eyes_in_img = face_tools.find_eyes(processed_img, [])