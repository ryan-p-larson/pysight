"""
PySight - Drawing Utilities
"""
from matplotlib.pyplot import imshow
import cv2 as cv


def cv2mpl(frame, color=False):
    """Helper function to convert a BGR img to RGB and display using Matplotlib.
    color (cv.COLOR_***)
        cv.COLOR_BGR2RGB for normal webcam photos
        cv.COLOR_BGR2GRAY for normal to grey
        cv.COLOR_GRAY2RGB for grey to image
    """
    final_frame = cv.cvtColor(frame, color) if color else frame
    fig = imshow(final_frame)
    return fig


def draw_circle(frame, x, y, radius=3):
    """Helper function to draw a circle on an image copy."""
    
    # Make sure we're in bounds of the image
    height, width = frame.shape[:2]
    assert 0 <= x <= width
    assert 0 <= y <= height

    # Copy and draw the circle
    img_copy = frame.copy()
    cv.circle(img_copy, (int(x), int(y)), radius, (255,255,255)) # White
    return img_copy


def draw_rectangle(frame, x, y, width, height):
    """Helper function to draw a rectangle on an image copy."""

    # Assert the rectangle is within image bounds
    img_height, img_width = frame.shape[:2]
    assert 0 <= x <= img_width
    assert 0 <= x+width <= img_width
    assert 0 <= y <= img_height
    assert 0 <= y+height <= img_height

    # Copy the image and draw the rect
    img_copy = frame.copy()
    cv.rectangle(img_copy, (int(x), int(y)), (int(x)+int(width), int(y)+int(height)), (255, 255, 255), 2)
    return img_copy