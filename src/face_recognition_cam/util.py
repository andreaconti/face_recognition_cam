"""
utility functions
"""

import os
import cv2
import numpy as np
import face_recognition_cam as fc


def load_known_faces(folder):
    faces_to_return = []
    names_to_return = []
    for f_name in os.listdir(folder):
        if f_name.lower().endswith(('.jpg', '.png')):
            img = cv2.cvtColor(cv2.imread(os.path.join(folder, f_name)), cv2.COLOR_BGR2RGB)
            faces = fc.crop_aligned_faces(img, resize=(112, 112))

            if len(faces) == 0:
                print(f'[WARNING] face not found in {f_name}, skipped.')
                continue
            if len(faces) > 1:
                print(f'[WARNING] too many faces found in {f_name}, skipped.')
                continue

            face = faces[0]
            name = os.path.basename(f_name)
            name = os.path.splitext(name)[0]
            names_to_return.append(name)

            faces_to_return.append(face)

    return names_to_return, np.array(faces_to_return)


def find_known(to_find, known, names, threshold):
    distances = np.sum(np.square(known - to_find), 1)
    idx = np.argmin(distances)
    if distances[idx] > threshold:
        return 'unknown'
    else:
        return names[idx]
