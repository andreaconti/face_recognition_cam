"""
Module containing person recognition
"""

from pkg_resources import resource_filename
import warnings
from sklearn.svm import SVC
import sklearn.utils
from sklearn.model_selection import cross_val_score
from typing import List
import numpy as np
from numpy import ndarray

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow.compat.v1 as tf


# resources


_meta_graph = resource_filename(
    'face_recognition_cam.resources.models',
    'mfn.ckpt.meta'
)

_weights = resource_filename(
    'face_recognition_cam.resources.models',
    'mfn.ckpt'
)


# CNN Embedder


class FaceEmbedder:
    """
    Class for face embedding
    """

    def __init__(self):
        self._session = tf.Session()
        saver = tf.train.import_meta_graph(_meta_graph)
        saver.restore(self._session, _weights)

    def embed_faces(self, faces: ndarray) -> ndarray:
        """
        Performs face embedding given a list of faces in a ndarray

        Parameters
        ----------
        faces: array_like
            faces parameter must have [N, 112, 112] shape

        Returns
        -------
        array_like
            of shape [N, 192] where each row is a face embedded
        """

        # load CNN endpoints
        images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

        # preprocess images
        faces = faces - 127.5
        faces = faces * 0.0078125

        faces_embedded = self._session.run(embeddings, feed_dict={
            images_placeholder: faces,
            phase_train_placeholder: False
        })

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
