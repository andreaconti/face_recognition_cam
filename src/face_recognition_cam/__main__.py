"""
face recognition cam main
"""


# IMPORTS


import argparse
import cv2
import face_recognition_cam as fc
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

    # parse and run
    args = parser.parse_args()
    args.func(vars(args))


# SUB-COMMAND FUNCTIONS


# embed command

def embed(args):

    recognizer = fc.FaceRecognizer()
    if os.path.isdir(args['DIR']):
        names, faces = fc.load_known_faces(args['DIR'])
        for name in names:
            print(f'OK: found face for {name}')
        faces_embedded = recognizer.embed_faces(faces)
        with open(args['o'], 'wb') as f:
            pickle.dump((names, faces_embedded), f)
        print(f'Saved into {args["o"]}')


# show command


def show(args):

    with fc.util.Camera() as cam:

        # load recognizer, archive and start camera and window
        recognizer = fc.FaceRecognizer()
        known_names, known_embedded = pickle.load(args['EMBED_FILE'])
        with fc.util.ImageWindow('camera') as window:
            while True:

                # load image from webcam
                frame = cam.image()
                if frame is None:
                    print('end of streaming')
                    break

                # find faces, and who they are
                faces, boxes = fc.crop_aligned_faces(frame, (112, 112), with_boxes=True)
                found_names = recognizer.assign_names(known_embedded, faces)

                # show image labeled
                window.show(frame, box_faces=(boxes, found_names))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# watch command


def watch(args):
    pass
