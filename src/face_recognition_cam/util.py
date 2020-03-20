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
        'shape_predictor_5_face_landmarks.dat'
    )
)


def _find_landmarks(img, res):
    shape = _landmarks_pred(img, res)
    landmark_points = np.empty(shape=(5, 2), dtype=int)
    for i, point in enumerate(shape.parts()):
        landmark_points[i, 0] = point.x
        landmark_points[i, 1] = point.y
    return landmark_points


def find_faces(img, landmarks=False, **kwargs):
    detector = dlib.get_frontal_face_detector()
    results = detector(img, 1)

    output = []
    for res in results:
        landmarks_ = None
        if landmarks:
            landmarks_ = _find_landmarks(img, res)
            if 'relative' in kwargs and kwargs['relative'] is True:
                landmarks_ = landmarks_ - np.array([res.left(), res.top()])

        output.append({
            'rectangle': ((res.left(), res.top()), (res.right(), res.bottom())),
            'landmarks': landmarks_
        })

    return output


def rotate_on_face(img, landmarks):

    # find eyes center
    right_eye = landmarks[0]
    left_eye = landmarks[2]
    eyes_center = (right_eye + left_eye) // 2

    # find angle
    rel_measures = right_eye - left_eye
    tangent = rel_measures[1] / rel_measures[0]
    angle = np.degrees(np.arctan(tangent))

    # rotate img
    rot_mat = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)
    aligned = cv2.warpAffine(
        img,
        rot_mat,
        tuple(img.shape[:2][::-1]),
        flags=cv2.INTER_LINEAR
    )

    # rotate landmarks
    landmarks_ = np.ones(shape=(landmarks.shape[0], landmarks.shape[1] + 1))
    landmarks_[:, :-1] = landmarks
    landmarks_ = np.round(np.matmul(rot_mat, landmarks_.T).T).astype(int)

    # find new bounding box
    eye_distance = landmarks_[0, 0] - landmarks_[2, 0]
    nose_distance = landmarks[4, 1] - landmarks_[0, 1]
    x1 = landmarks_[2, 0] - int(eye_distance*0.30)
    x2 = landmarks_[0, 0] + int(eye_distance*0.30)
    y1 = landmarks_[0, 1] - int(nose_distance*0.7)
    y2 = landmarks_[4, 1] + int(nose_distance*1.3)

    return aligned, landmarks_, ((x1, y1), (x2, y2))
