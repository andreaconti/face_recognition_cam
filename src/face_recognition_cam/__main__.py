"""
face recognition cam main
"""

# Imports

import argparse
import cv2
import face_recognition_cam as fc


# Arguments parsing

parser = argparse.ArgumentParser('face detection camera')
parser.add_argument('known_faces', help='path to a folder with an image of each known person, the file name is their name')
args = vars(parser.parse_args())


# Main


def main():
    with fc.util.Camera() as cam:

        # load recognizer
        recognizer = fc.FaceRecognizer()

        # embed known people
        known_names, known_faces = fc.load_known_faces(args['known_faces'])
        known_embedded = recognizer.embed_faces(known_faces)

        # start monitor camera
        with fc.util.ImageWindow('camera') as window:
            while True:

                # load image from webcam
                frame = cam.image()
                if frame is None:
                    print('end of streaming')
                    break

                # find faces and preprocess
                faces, boxes = fc.crop_aligned_faces(frame, (112, 112), with_boxes=True)

                # find who they are
                found_names = recognizer.assign_names(known_embedded, faces)

                # show image
                window.show(frame, box_faces=(boxes, found_names))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
