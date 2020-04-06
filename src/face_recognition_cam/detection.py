"""
Functions for face detection and landmarks
"""

import dlib
import numpy as np
from numpy import ndarray
import cv2
from typing import Tuple
from pkg_resources import resource_filename

# FACE DETECTION

_landmarks_pred = dlib.shape_predictor(
    resource_filename(
        'face_recognition_cam.resources.models',
        'shape_predictor_5_face_landmarks.dat'
    )
)


def find_face_boxes(img: ndarray) -> ndarray:
    """
    Finds all faces inside the image using HOG and SVM dlib implementation.
    Faces boxes are returned in a numpy ndarray.

    Parameters
    ----------
    img : array_like
        gray-scale or colored image in which search faces

    Returns
    -------
    array_like
        each row contains left, top, right, bottom coordinates of each box
        found
    """
    detector = dlib.get_frontal_face_detector()
    results = detector(img, 0)

    output = []
    for res in results:
        output.append([res.left(), res.top(), res.right(), res.bottom()])

    return np.array(output)


def find_5_landmarks(img: ndarray, box: Tuple[int, int, int, int]) -> ndarray:
    """
    Search for 5 landmarks in a face contained in `img` and localized
    by `box`.

    Parameters
    ----------
    img : array_like
        gray-scale or colored image in which search faces
    box : tuple
        contains left, top, right, bottom integer coordinates of box

    Returns
    -------
    array_like
        each row contains x and y coordinates of one of the 5 landmarks
        detected
    """
    left, top, right, bottom = box
    shape = _landmarks_pred(img, dlib.rectangle(left, top, right, bottom))
    landmark_points = np.empty(shape=(5, 2), dtype=int)
    for i, point in enumerate(shape.parts()):
        landmark_points[i, 0] = point.x
        landmark_points[i, 1] = point.y
    return landmark_points


def align_face(img: ndarray, landmarks_5: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Image is rotated around provided landmarks in order align face,
    returns the aligned image, new landmarks and new box

    Parameters
    ----------
    img : array_like
        original gray-scale or colored image
    landmarks_5 : array_like
        position of 5 face landmarks reshaped as a 5 rows and 2 columns
        matrix


    Returns
    -------
    array_like, array_like, array_like
        respectively aligned image, landmarks and box
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
    x1 = max(landmarks_[2, 0] - int(eye_distance*0.30), 0)
    x2 = min(landmarks_[0, 0] + int(eye_distance*0.30), aligned.shape[1])
    y1 = max(landmarks_[0, 1] - int(nose_distance*0.7), 0)
    y2 = min(landmarks_[4, 1] + int(nose_distance*1), aligned.shape[0])

    return aligned, landmarks_, np.array([x1, y1, x2, y2])


def crop_aligned_faces(img: ndarray, resize: Tuple[int, int], with_boxes: bool = False):
    """
    Search for all faces in `img` and returns them resized with the specified
    `resize` and aligned using 5 landmarks. If you want also face boxes in original img
    use `with_boxes` option.

    Parameters
    ----------
    img : array_like
        original gray-scale or colored image
    resize : (int, int)
        height and width size of the cropped faces returned
    with_boxes : bool
        if return also faces boxes coordinates in the original image


    Returns
    -------
    array_like
        if with_boxes = False, list of faces cropped and aligned

    array_like, array_like
        if with_boxes = True, list of faces cropped and aligned and boxes found in `img`
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
