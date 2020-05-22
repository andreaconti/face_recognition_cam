"""
face recognition cam main
"""


# IMPORTS


import argparse
import cv2
import face_recognition_cam as fc
import numpy as np
import os
import pickle


# ARGUMENT PARSING & MAIN


def main():

    parser = argparse.ArgumentParser('face detection camera')
    subparsers = parser.add_subparsers(help='sub-command help')

    # embed command
    parser_embed = subparsers.add_parser('embed', help='embed a folder of faces in known faces')
    parser_embed.add_argument('DIR', type=str, help='directory containing images of known people, file names are used')
    parser_embed.add_argument('-o', type=str, default='archive.pkl', help='path to output embedded file')
    parser_embed.set_defaults(func=embed)

    # watch command
    parser_watch = subparsers.add_parser('watch', help='starts to watch from a cam')
    parser_watch.add_argument('EMBED_FILE', type=argparse.FileType('rb'), help='path to embedded known people')
    parser_watch.set_defaults(func=watch)

    # show command
    parser_show = subparsers.add_parser('show', help='watch from cam and show a window with labeled faces, for fun')
    parser_show.add_argument('EMBED_FILE', type=argparse.FileType('rb'), help='path to embedded known people')
    parser_show.set_defaults(func=show)

    # recognize command
    parser_test = subparsers.add_parser('recognize', help='test on a dataset')
    parser_test.add_argument('EMBED_FILE', type=argparse.FileType('rb'), help='path to embedded known people')
    parser_test.add_argument('IMG', type=str, help='path to an image')
    parser_test.set_defaults(func=recognize)

    # parse and run
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(vars(args))
    else:
        parser.print_help()


# SUB-COMMAND FUNCTIONS


# embed command

def embed(args):

    recognizer = fc.FaceRecognizer()
    if os.path.isdir(args['DIR']):

        # extract faces
        people = fc.load_faces(args['DIR'])

        for name, faces in people.items():
            print(f'faces found for {name}: {len(faces)}')

        # embedding
        dataset = recognizer.generate_dataset(people)

        # save
        with open(args['o'], 'wb') as f:
            pickle.dump(dataset, f)
        print(f'Saved into {args["o"]}')
    else:
        print('error: not dir')


# show command

def show(args):

    with fc.util.Camera() as cam:

        # load recognizer, archive and start camera and window
        detector = fc.FaceDetector()
        recognizer = fc.FaceRecognizer()
        dataset = pickle.load(args['EMBED_FILE'])
        with fc.util.ImageWindow('camera') as window:
            while True:

                # load image from webcam
                frame = cam.image()
                if frame is None:
                    print('end of streaming')
                    break

                # find faces, and who they are
                faces, boxes = detector.crop_aligned_faces(frame, (112, 112), with_boxes=True)
                if len(faces) != 0:
                    found_names = recognizer.assign_names(dataset, faces)
                    found_names = ['{} ({:.2f} %)'.format(name, conf*100) for name, conf in found_names]
                    window.show(frame, box_faces=(boxes, found_names))
                else:
                    window.show(frame)

                # check for end
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# watch command


def watch(args):
    pass


# recognize command

def recognize(args):

    detector = fc.FaceDetector()
    embedder = fc.FaceEmbedder()
    recognizer = pickle.load(args['EMBED_FILE'])

    img = cv2.imread(args['IMG'])
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.crop_aligned_faces(img, resize=(112, 112))

    faces_embedded = embedder.embed_faces(faces)
    names = recognizer.assign_names(faces_embedded)

    for name in names:
        print(f'found: {name}')
