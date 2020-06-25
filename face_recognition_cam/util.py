"""
utility functions
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray

import face_recognition_cam as fc

# FILES HANDLING UTILS


def load_faces(folder: str) -> Dict[str, ndarray]:
    """
    Search for faces in files of type jpg, png and mp4 inside `folder` path. Images or
    frames of videos with none or more than one face are skipped.
    To each recognized face is assigned a name equal to the name of the file until
    '-' character. For instance a file named 'andreaconti-1.jpg' will led to
    'andreaconti' label. If a video is found, frames are sampled at 1s distance.

    Parameters
    ----------
    folder : str
        path of the folder in which search for faces

    Returns
    -------
    array_like, array_like
        tuple containing a list of faces and the list of matching labels
    """

    detector = fc.FaceDetector()
    result: Dict[str, ndarray] = {}

    for f_name in os.listdir(folder):

        # assigned name
        name = os.path.basename(f_name)
        name = os.path.splitext(name)[0]
        name = name.split("-")[0]

        # load face from image
        if f_name.lower().endswith((".jpg", ".png")):
            img = cv2.cvtColor(
                cv2.imread(os.path.join(folder, f_name)), cv2.COLOR_BGR2RGB
            )
            faces = detector.crop_aligned_faces(img, resize=(112, 112))

            if len(faces) == 0:
                print(f"[WARNING] face not found in {f_name}, skipped.")
                continue
            if len(faces) > 1:
                print(f"[WARNING] too many faces found in {f_name}, skipped.")
                continue

            if name not in result.keys():
                result[name] = []
            face = faces[0]
            result[name].append(face)

        # load face from video
        elif f_name.lower().endswith(".mp4"):
            faces = _load_from_video(os.path.join(folder, f_name))

            if name not in result.keys():
                result[name] = []
            face = faces[0]
            result[name].extend(faces)

    for k in result.keys():
        result[k] = np.stack(result[k])
    return result


def _load_from_video(video_path):
    detector = fc.FaceDetector()
    faces_to_return = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    i = 0
    while ret is True:
        if i % frame_rate == 0:
            faces = detector.crop_aligned_faces(frame, resize=(112, 112))
            if len(faces) == 1:
                faces_to_return.append(faces[0])
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1

    cap.release()
    return faces_to_return


# VIDEO HANDLING


class Camera:
    """
    It loads the main system camera and provides an easy way to
    retrieve frames.

    Example
    -------

        >>> with Cameras as cam:
        ...     img = cam.image()  # returns the rgb frame

    """

    def __init__(self):
        self._cap = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(0)
        return self

    def image(self):
        ret, img = self._cap.read()
        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None

    def __exit__(self, type, value, traceback):
        self._cap.release()


class ImageWindow:
    """
    Provides an easy way to create a window to show an image and
    optionally sorround faces with labeled boxes.

    Attributes
    ----------
    name: str
        a name for the window (is used as an identifier under the
        hood for the window)

    Example
    -------

        >>> with ImageWindow('my_stream') as window:
        ...     img = np.random.randn(256, 512, 3).astype(np.uint8)
        ...     window.show(img)

    """

    def __init__(self, name):
        self._name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    def show(
        self,
        img: ndarray,
        box_faces: Optional[Tuple[List[Tuple[int, int, int, int]], List[str]]] = None,
    ):
        """
        show the provided image and if provided sorround faces with a labeled
        box

        Parameters
        ----------
        img: ndarray
            the image to be showed.
        box_faces: (face_boxes, names)
            where face_boxes is a list of (x1, y1, x2, y2) coordinates and names
            is a list of matching names.

        Examples
        --------

            >>> window = ImageWindow('window')
            >>> img = np.random.randn(256, 512, 3).astype(np.uint8)
            >>> boxes = ([[50, 50, 100, 100]], ['my_face'])
            >>> window.show(img, boxes)

        """

        if box_faces is not None:
            img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for box, name in zip(*box_faces):
                x1, y1, x2, y2 = box
                cv2.rectangle(img_, (x1, y1), (x2, y2), (80, 18, 236), 2)
                cv2.rectangle(img_, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img_, name, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

            cv2.imshow(self._name, img_)
        else:
            cv2.imshow(self._name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def close(self):
        """
        destroy the window
        """
        cv2.destroyWindow(self._name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        cv2.destroyWindow(self._name)


# Camera alert


class CameraAlert:
    """
    Utility class to trigger functions execution when a specific face is
    recognized from camera.

    Example
    -------

    >>> alert = CameraAlert()
    >>> @alert.register("andrea")
    ... def on_andrea():
    ...     print("Found andrea!")
    >>> alert.watch(dataset)
    """

    def __init__(self):
        self._handlers = {}

    def register(self, name: str):
        def _register(fn):
            if name not in self._handlers:
                self._handlers[name] = []
            self._handlers[name].append(fn)
            return fn

        return _register

    def watch(self, dataset: Dict[str, ndarray]):
        detector = fc.FaceDetector()
        recognizer = fc.FaceRecognizer()

        with Camera() as cam:
            while True:

                frame = cam.image()
                if frame is None:
                    return

                # find faces, and who they are
                faces = detector.crop_aligned_faces(frame, (112, 112))
                if len(faces) != 0:
                    found_names = recognizer.assign_names(dataset, faces)

                    # call registered functions
                    for name, _ in found_names:
                        if name in self._handlers:
                            for fn in self._handlers[name]:
                                fn()
