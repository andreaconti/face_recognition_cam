"""
Utils functions used in camera pipeline
"""

import dlib
import numpy as np
import cv2
from pkg_resources import resource_filename

# load predictor with module

_landmarks_pred = dlib.shape_predictor(
    resource_filename(
        'face_recognition_cam.resources.models',
        'shape_predictor_68_face_landmarks.dat'
    )
)


def _find_landmarks(img, res):
    shape = _landmarks_pred(img, res)
    landmark_points = np.empty(shape=(68, 2), dtype=int)
    for i, point in enumerate(shape.parts()):
        landmark_points[i, 0] = point.y
        landmark_points[i, 1] = point.x
    return landmark_points


def find_faces(img, landmarks=False, **kwargs):
    detector = dlib.get_frontal_face_detector()
    results = detector(img, 1)

    output = []
    for res in results:
        if landmarks:
            landmarks_ = _find_landmarks(img, res)
            rmin, rmax = landmarks_[:, 0].min(), landmarks_[:, 0].max()
            cmin, cmax = landmarks_[:, 1].min(), landmarks_[:, 1].max()

            if 'relative' in kwargs and kwargs['relative'] is True:
                landmarks_ = landmarks_ - np.array([rmin, cmin])

            output.append({
                'rectangle': ((rmin, cmin), (rmax, cmax)),
                'landmarks': landmarks_
            })
        else:
            output.append({
                'rectangle': ((res.top(), res.left()), (res.bottom(), res.right())),
                'landmarks': None
            })

    return output


_landmark_standard = None


def warp_face(face, landmarks):

    # load _landmark_standard points if not loaded
    global _landmark_standard
    if _landmark_standard is None:
        name = resource_filename('face_recognition_cam.resources.data', 'dlib-landmark-mean.csv')
        _landmark_standard = np.genfromtxt(name, delimiter=',')

    # adapt standard_landmarks
    rows, cols = face.shape
    custom_std_landmarks = _landmark_standard * np.array([rows/200, cols/200])  # 200 size of std landmarks img
    h, status = cv2.findHomography(landmarks, custom_std_landmarks)
    warped = cv2.warpPerspective(face, h, (cols, rows))

    return warped, custom_std_landmarks
