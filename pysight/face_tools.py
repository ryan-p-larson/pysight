"""
PySight - Face

Module implementing facial alignment for computer vision. Uses Haar Cascades 
to find faces in photos (less accurate but quicker than Dlib's histogram of
gradients), and uses Dlibs AAM.
"""

import numpy as np
import dlib
import cv2


# Active Shape Model template
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


class FaceTools(object):
    """Object containing OpenCV cascades and dlib facial landmark clfs."""

    #: Landmark indices.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, face_f, eyes_f, dlib_f):
        """
        Instantiates the object with all the clfs.
        Args:
            face_f (str): Path to (Haar_*.xml) face cascade file from OpenCV.
            eyes_f (str): Path to (Haar_*.xml) eye cascade file from OpenCV.
            dlib_f (str): Path to Dlib's 68 facial landmarks file (landmarks_*.dat).
        """
        self.face_clf = cv2.CascadeClassifier(face_f)
        self.eyes_clf = cv2.CascadeClassifier(eyes_f)
        self.dlib_clf = dlib.shape_predictor(dlib_f)


    def find_faces(self, img, multi=False):
        """
        Function to find faces in an image using Haar Cascades.
        Args:
            img (numpy.ndarray): Image to process.
        Returns:
            All facial bounding boxes within 'img'.
        """
        assert img is not None
        try:
            return self.face_clf.detectMultiScale(
                    image=img, 
                    #scaleFactor, 
                    minNeighbors=5, 
                    flags=cv2.CASCADE_SCALE_IMAGE, 
                    minSize=(30, 30)
                    #maxSize
                )
        except Exception as e:
            print ("Warning: {}".format(e))
            return ()


    def find_eyes(self, img, face_rect=[]):
        """
        Function to find the bounding box of eyes in a webcam image.
        Args:
            img (numpy.ndarray): Image to process.
            face_rect (???): Bounding box of face
        Returns:
            Eyes' bounding boxes within 'img'. () if empty
        """
        assert img is not None
        try:
            return self.eyes_clf.detectMultiScale(
                    image=img, 
                    scaleFactor=1.03, 
                    minNeighbors=5, 
                    flags=0, 
                    minSize=(30, 30) 
                    #maxSize
                )
        except Exception as e:
            print ("Warning: {}".format(e))
            return ()


    def find_landmarks(self, img, face_rect):
        """
        Function to identify 68 facial landmarks from a given face img.
        Args:
            img (np.ndarray): Input image
            face_rect (list): List of x1, x2, y1, y2 points bounding the face.
        Returns:
            landmarks (list[points]): List of 68 facial landmarks if successful, else []
        """
        assert img is not None
        # If we don't have a bounding box then we're SOL
        if len(face_rect) < 1: return []

        # Convert face_rect list to Dlib rectangle for dlib.shape_predictor().
        dlib_rect = dlib.rectangle(
                int(face_rect[0][0]),
                int(face_rect[0][1]),
                int(face_rect[0][2]),
                int(face_rect[0][3])
            )

        try:
            landmarks_dlib = self.dlib_clf(img, dlib_rect)

            # Convert to NP formatted points afterward
            # TO-DO

            return landmarks_dlib
        except Exception as e:
            print ("Warning: {}".format(e))
            return []