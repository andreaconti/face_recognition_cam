# Face Recognition Cam

This project aims to build the software for a camera able to recognize movements in the observed scene, search for
faces in it and if it is found recognize it. If it is not a known person trigger an alarm.

In order to do that the camera finds regions of interest through movements, then the following process is followed:

1. find faces
2. find facial landmarks
3. face are warped to standard landmark positions
4. through a CNN the face is encoded in a vector
5. search of similarities into a database of known faces
6. if it is not found an alarm is triggered

