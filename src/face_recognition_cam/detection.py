"""
Functions for face detection and landmarks
"""

import dlib
import numpy as np
import cv2
from pkg_resources import resource_filename

# FACE DETECTION

_landmarks_pred = dlib.shape_predictor(
    resource_filename(
        'face_recognition_cam.resources.models',
        'shape_predictor_5_face_landmarks.dat'
    )
)


def find_face_boxes(img):
    """
    Returns a 2D array, each row contains left, top, right, bottom
    coordinates of face boxes find in the image
    """
    detector = dlib.get_frontal_face_detector()
    results = detector(img, 1)

    output = []
    for res in results:
        output.append([res.left(), res.top(), res.right(), res.bottom()])

    return np.array(output)


def find_5_landmarks(img, box):
    """
    Returns a 2D array, each row contains a landmark contained in the
    given box
    """
    left, top, right, bottom = box
    shape = _landmarks_pred(img, dlib.rectangle(left, top, right, bottom))
    landmark_points = np.empty(shape=(5, 2), dtype=int)
    for i, point in enumerate(shape.parts()):
        landmark_points[i, 0] = point.x
        landmark_points[i, 1] = point.y
    return landmark_points


def align_face(img, landmarks_5):
    """
    Image is rotated around provided landmarks in order align face,
    returns the aligned image, new landmarks and new box
    """

    # find eyes center
    right_eye = landmarks_5[0]
    left_eye = landmarks_5[2]
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
    landmarks_ = np.ones(shape=(landmarks_5.shape[0], landmarks_5.shape[1] + 1))
    landmarks_[:, :-1] = landmarks_5
    landmarks_ = np.round(np.matmul(rot_mat, landmarks_.T).T).astype(int)

    # find new bounding box
    eye_distance = landmarks_[0, 0] - landmarks_[2, 0]
    nose_distance = landmarks_[4, 1] - landmarks_[0, 1]
    x1 = landmarks_[2, 0] - int(eye_distance*0.30)
    x2 = landmarks_[0, 0] + int(eye_distance*0.30)
    y1 = landmarks_[0, 1] - int(nose_distance*0.7)
    y2 = landmarks_[4, 1] + int(nose_distance*1)

    return aligned, landmarks_, np.array([x1, y1, x2, y2])


def crop_aligned_faces(img, resize, with_boxes=False):
    """
    Returns all faces in the image, aligned and cropped, optionally resized.
    """

    if len(img.shape) != 3 and len(img.shape) != 2:
        raise ValueError('img must have shape 2 or 3')

    shape = np.array([*img.shape])
    shape[0] = resize[0]
    shape[1] = resize[1]

    boxes = find_face_boxes(img)
    landmarks = [find_5_landmarks(img, box) for box in boxes]
    aligned_faces = [align_face(img, landmarks_) for landmarks_ in landmarks]

    faces = []
    for (aligned, _, box) in aligned_faces:
        left, top, right, bottom = box
        faces.append(cv2.resize(aligned[top:bottom, left:right], tuple(resize)))

    if with_boxes:
        return np.array(faces), boxes
    else:
        return np.array(faces)
