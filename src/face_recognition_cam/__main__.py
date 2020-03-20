"""
face recognition cam main
"""

# Imports


import os
import argparse
import cv2
import face_recognition_cam as fc

import numpy as np
from pkg_resources import resource_filename
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Arguments parsing

parser = argparse.ArgumentParser('face detection camera')
parser.add_argument('known_faces', help='path to a folder with an image of each known person, the file name is their name')
args = vars(parser.parse_args())


# Resources


_meta_graph = resource_filename(
    'face_recognition_cam.resources.models',
    'mfn.ckpt.meta'
)

_weights = resource_filename(
    'face_recognition_cam.resources.models',
    'mfn.ckpt'
)


# Util functions


def load_known_faces(folder):
    faces_to_return = []
    names_to_return = []
    for f_name in os.listdir(folder):
        if f_name.lower().endswith(('.jpg', '.png')):
            img = cv2.cvtColor(cv2.imread(os.path.join(folder, f_name)), cv2.COLOR_BGR2RGB)
            faces = fc.util.find_faces(img, landmarks=True)

            if len(faces) == 0:
                print(f'[WARNING] face not found in {f_name}, skipped.')
                continue
            if len(faces) > 1:
                print(f'[WARNING] too many faces found in {f_name}, skipped.')
                continue

            face = faces[0]
            rectified, _, ((x1, y1), (x2, y2)) = fc.util.rotate_on_face(img, landmarks=face['landmarks'])
            rectified = cv2.resize(rectified[y1:y2, x1:x2], (112, 112))
            rectified = rectified - 127.5
            rectified = rectified * 0.0078125

            name = os.path.basename(f_name)
            name = os.path.splitext(name)[0]
            names_to_return.append(name)

            faces_to_return.append(rectified)

    return names_to_return, faces_to_return


def find_known(to_find, known, names, threshold):
    distances = np.sum(np.square(known - to_find), 1)
    idx = np.argmin(distances)
    if distances[idx] > threshold:
        return 'unknown'
    else:
        return names[idx]


# Main

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)

    with tf.Graph().as_default():
        with tf.Session() as sess:

            # load CNN and endpoints
            saver = tf.train.import_meta_graph(_meta_graph)
            saver.restore(sess, _weights)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            # embed known people
            known_names, known_faces = load_known_faces(args['known_faces'])
            known_embedded = sess.run(embeddings, feed_dict={
                images_placeholder: known_faces,
                phase_train_placeholder: False
            })

            # start monitor camera
            while True:

                # load image from webcam
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # find faces and preprocess
                faces = fc.util.find_faces(frame, landmarks=True)

                faces_ = []
                for face in faces:

                    # preprocess and append
                    rectified, _, ((x1, y1), (x2, y2)) = fc.util.rotate_on_face(frame, landmarks=face['landmarks'])
                    rectified = cv2.resize(rectified[y1:y2, x1:x2], (112, 112))
                    rectified = rectified - 127.5
                    rectified = rectified * 0.0078125
                    faces_.append(rectified)

                # find who they are
                if len(faces_) != 0:
                    feed_dict = {images_placeholder: np.array(faces_), phase_train_placeholder: False}
                    faces_embedded = sess.run(embeddings, feed_dict=feed_dict)
                    found_names = [find_known(face_embed, known_embedded, known_names, 1) for face_embed in faces_embedded]

                # plot
                for i, face in enumerate(faces):

                    # Draw rectangle and label with a name below the face
                    (x1, y1), (x2, y2) = face['rectangle']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                    cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, found_names[i], (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

                # show image
                cv2.imshow('webcam', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
