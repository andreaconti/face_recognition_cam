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
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('webcam', cv2.WINDOW_AUTOSIZE)

    recognizer = fc.FaceRecognizer()

    # embed known people
    known_names, known_faces = fc.load_known_faces(args['known_faces'])
    known_embedded = recognizer.embed_faces(known_faces)

    # start monitor camera
    while True:

        # load image from webcam
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # find faces and preprocess
        faces, boxes = fc.crop_aligned_faces(frame, (112, 112), with_boxes=True)

        # find who they are
        found_names = recognizer.assign_names(known_embedded, faces)

        # plot
        for i, box in enumerate(boxes):

            # Draw rectangle and label with a name below the face
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, found_names[i], (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

        # show image
        cv2.imshow('webcam', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
