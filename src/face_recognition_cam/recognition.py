"""
Module containing person recognition
"""

from pkg_resources import resource_filename
from typing import List, Dict, Tuple
import numpy as np  # type: ignore
from numpy import ndarray
import mxnet as mx  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore


class FaceRecognizer:
    """
    Class for face embedding and recognizing
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

        # embed
        faces = np.moveaxis(faces, -1, 1)
        faces_embedded = self._model(mx.nd.array(faces))
        faces_embedded = faces_embedded.asnumpy()

        return faces_embedded

    def generate_dataset(self, people: Dict[str, ndarray]) -> Dict[str, ndarray]:
        """
        Takes as input a dictionary containing as key the name of each person and as value
        a ndarray representing a batch of images of that person and returns another
        dictionary 'name': embedding.

        Parameters
        ----------
        people: Dict[str, ndarray]
            a dictionary containing for each name a ndarray containing images of that
            person. each ndarray must be of shape [N, 112, 112, 3]

        Returns
        -------
        Dict[str, ndarray]
            where each ndarray is the embedding
        """
        result: Dict[str, ndarray] = {}
        for name, imgs in people.items():
            embeddings = self.embed_faces(imgs)
            result[name] = np.mean(embeddings, axis=0)
        return result

    def assign_names(self, dataset: Dict[str, ndarray], faces: ndarray,
                     min_confidence: float = 0.6) -> List[Tuple[str, float]]:
        """
        Assign a name to each face in `faces`.

        Parameters
        ----------
        dataset: Dict[str, ndarray]
            a dictionary in which each name is associated to an embedding. It can be
            generated with `generate_dataset` method.

        faces: ndarray
            a numpy ndarray of shape [N, 112, 112, 3] where each [112, 112, 3] is a
            face.

        min_confidence: float, default 0.6
            if among people the maximum found confidence is less than `min_confidence`
            such face is labeled as 'unknown'

        Returns
        -------
        List[str]
            the name associated to each face, can be 'unknown' if the maximum confidence
            found is less than `min_confidence`
        """
        names, data_emb = zip(*dataset.items())
        names, data_emb = np.array(names), np.stack(data_emb)
        faces_emb = self.embed_faces(faces)
        confidence_matrix = 1 - cdist(faces_emb, data_emb, metric='cosine')
        best = np.argmax(confidence_matrix, axis=0)
        confidences = confidence_matrix[np.arange(confidence_matrix.shape[0]), best]
        names = names[best]
        names[confidences < min_confidence] = 'unknown'
        result = list(zip(names, confidences))
        return result
