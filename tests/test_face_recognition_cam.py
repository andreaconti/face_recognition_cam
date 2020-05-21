"""
Test of simple face detection
"""

from face_recognition_cam import __version__
import face_recognition_cam as fc
import cv2
import os


def test_version():
    assert __version__ == '0.1.0'


file_path = os.path.join(os.path.dirname(__file__), 'test_data', 'sample.jpg')


def test_face_detection():

    # load img
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find faces
    boxes = fc.find_face_boxes(img)
    assert boxes.shape[0] == 1


def test_face_embedding():

    # load img
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find faces
    faces = fc.crop_aligned_faces(img, resize=(112, 112))
    assert faces.shape[0] == 1

    # embed
    embedder = fc.FaceEmbedder()
    faces_emb = embedder.embed_faces(faces)
    assert faces_emb.shape == (1, 128)
