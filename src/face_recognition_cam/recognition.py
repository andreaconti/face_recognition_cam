"""
Module containing person recognition
"""

from pkg_resources import resource_filename
import warnings
from sklearn.svm import SVC, OneClassSVM
import numpy as np

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


# Recognizer


class FaceEmbedder:

    def __init__(self):
        self._session = tf.Session()
        saver = tf.train.import_meta_graph(_meta_graph)
        saver.restore(self._session, _weights)

    def embed_faces(self, faces):

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


class FaceRecognizer:

    def __init__(self):
        self._recognizer = SVC(probability=True)
        self._outlier = OneClassSVM(nu=0.01)

    def fit(self, known_embed, names):
        self._recognizer.fit(known_embed, names)
        self._outlier.fit(known_embed)

    def assign_names(self, embedded_faces):
        scores = self._recognizer.predict_proba(embedded_faces)
        names = self._recognizer.classes_[np.argmax(scores, 1)]

        unknowns = self._outlier.predict(embedded_faces)
        names[unknowns == -1] = 'unknowns'

        return names
