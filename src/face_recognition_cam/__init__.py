"""
Face Recognition Cam Package
"""

__version__ = '0.1.0'


from .detection import (
    find_face_boxes,
    find_5_landmarks,
    align_face,
    crop_aligned_faces
)

from . import util
from .util import (
    load_faces
)

from .recognition import (
    FaceRecognizer,
    FaceEmbedder
)
