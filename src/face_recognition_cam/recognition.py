"""
Module containing person recognition
"""

from pkg_resources import resource_filename
from sklearn.svm import SVC  # type: ignore
import sklearn.utils  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
from typing import List
import numpy as np  # type: ignore
from numpy import ndarray
import mxnet as mx  # type: ignore


# CNN Embedder


class FaceEmbedder:
    """
    Class for face embedding
    """

    def __init__(self):

        # find file path
        params_path = resource_filename(
            'face_recognition_cam.resources.models',
            'mobileFaceNet-0000.params'
        )
        symbols_path = resource_filename(
            'face_recognition_cam.resources.models',
            'mobileFaceNet-symbol.json'
        )

        # load model
        ctx = mx.cpu()
        sym = mx.sym.load_json(open(symbols_path, 'r').read())
        model = mx.gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))
        model.load_parameters(params_path, ctx=ctx)
        self._model = model

    def embed_faces(self, faces: ndarray) -> ndarray:
        """
        Performs face embedding given a list of faces in a ndarray.

        Parameters
        ----------
        faces: array_like
            faces parameter must have [N, 112, 112, 3] shape, images must be RGB.

        Returns
        -------
        array_like
            of shape [N, 192] where each row is a face embedded
        """
        if len(faces.shape) == 3:
            faces = faces[None, :, :, :]
        elif len(faces.shape) == 4:
            _, h, w, c = faces.shape
            if c != 3 or h != 112 or w != 112:
                raise ValueError('expected images of shape 3x112x112')
        else:
            raise ValueError('shape must be 3 or 4 (a batch)')

        # preprocess images
        faces = np.moveaxis(faces, -1, 1)
        faces = faces[:, ::-1, :, :]  # RGB -> BGR
        faces = faces - 127.5
        faces = faces * 0.0078125

        # embed
        faces_embedded = self._model(mx.nd.array(faces))
        faces_embedded = faces_embedded.asnumpy()

        return faces_embedded


# Instances of the recognizer

_unknowns = resource_filename(
    'face_recognition_cam.resources.data',
    'unknowns_db.csv'
)


class FaceRecognizer:

    def __init__(self):
        self._recognizer = SVC()

    def fit(self, known_embed: ndarray, names: ndarray) -> float:
        """
        trains the FaceRecognizer using a list of examples composed by
        `known_embed` and `names`.

        Parameters
        ----------
        known_embed : array_like
            array of shape [N, 192] examples of embedded faces

        names : array_like of str labels
            array of N elements with `str` labels, label 'unknown'
            must not be contained in this list

        Returns
        -------
        mean_score : float
            returns mean accuracy performed on a 5 cross validation on
            training set
        """

        # load unknowns
        unknowns = np.loadtxt(_unknowns, delimiter=',')
        unknowns = unknowns[:len(known_embed) // len(np.unique(names))]
        X = np.vstack([unknowns, known_embed])
        y = np.hstack([
            np.array(['unknown'] * len(unknowns)),
            np.array(names)
        ])
        X, y = sklearn.utils.shuffle(X, y)

        # fitting
        scores = cross_val_score(self._recognizer, X, y, cv=5)
        mean_score = np.mean(scores)
        self._recognizer.fit(X, y)

        return mean_score

    def assign_names(self, embedded_faces: ndarray) -> List[str]:
        """
        Assigns names to `embedded_faces`

        Parameters
        ----------
        embedded_faces : array_like
            array_like of shape [N, 192]

        Returns
        -------
        list of strings
            assigned name to each example, between labels there is also
            'unknown' if the face is not between faces in the training
            set.

        """
        names = self._recognizer.predict(embedded_faces)
        return names
