"""
Face Recognition Cam Package
"""

__version__ = "0.1.0"

__all__ = ["util", "FaceDetector", "FaceRecognizer", "load_faces"]

from . import util
from .detection import FaceDetector
from .recognition import FaceRecognizer
from .util import load_faces
