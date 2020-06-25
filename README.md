# Face Recognition Cam

This project aims to implement a simple API for a camera with face recognition features mainly for didactic purposes but
I hope it'll be useful to someone.

To accomplish this task some steps are required:

1. find faces
2. find facial landmarks
3. warp faces to standard landmark positions
4. encode faces into vectors
5. search of similarities into a database of known faces

`face_recognition_cam` takes takes care of all of them.

## API Usage

`face_recognition_cam` can be used as a python library, at the core of the library there are two main classes:

- `FaceDetector`, used to detect a faces in the image, crop them, identify facial landmarks and align faces
- `FaceRecognizer`, used to assign a label to each known face or classify them as unknown

Finally some utilities are contained into `util` module:

- `Camera` provides a way to retrieve frames from a camera.
- `CameraAlert` class can be used to monitor a camera and execute custom functions when a face is recognized.

### FaceDetector

![sample image](readme_files/sample.jpg)

```python
import face_recognition_cam as fc

# here we detect all faces contained in the image, this function takes care to:
# - find each face
# - find 5 facial landmarks and use them to align the face
# - resize each face
# - stack them into a numpy ndarray of shape N x H x W x C
face_detector = fc.FaceDetector()
faces = face_detector.crop_aligned_faces(img, resize=(250, 250))
```

![sample_crop_face](readme_files/sample_crop_face.jpg)

### FaceRecognizer

This class takes care to label each face with a name, to do that a dataset of known faces is needed, such dataset can be
generated using `generate_dataset`. `assign_names` method is then used to label faces.

```python
import face_recognition_cam as fc

recognizer = fc.FaceRecognizer()

# dataset generation, people is a dictionary containing as key names and as value a
# set of N faces of that person stacked into an ndarry of shape N x 112 x 112 x 3.
# In `util` module see `load_faces` function to generate people dictionary.

people = {
    "andrea": np.random.randn(3, 112, 112, 3).astype(np.uint8)
}
dataset = recognizer.generate_dataset(people)

# then is possible to identify faces, `min_confidence` is used to choose a threshold
# for unknown labeling

min_confidence = 0.7
names = recognizer.assign_names(dataset, faces, min_confidence)
```

### CameraAlert

This class is useful when you simply want to execute some code when a face is recognized.

```python
import face_recognition_cam as fc

alert = fc.util.CameraAlert()

@alert.register("andrea"):
def say_hello():
    print("Welcome Andrea!")

alert.watch(dataset)
```

## CLI Usage

A simple CLI interface is also available through the `facecam` tool.

```bash
$ facecam
usage: face detection camera [-h] {embed,trigger,show,recognize} ...

positional arguments:
  {embed,trigger,show,recognize}
                        sub-command help
    embed               embed a folder of faces in known faces
    trigger             trigger from the camera and call scripts when
                        recognize someone
    show                watch from cam and show a window with labeled faces,
                        for fun and debug
    recognize           writes the name of all recognized people in the
                        provided image

optional arguments:
  -h, --help            show this help message and exit
```

`facecam embed` can be used to generate the dataset and `facecam show` can be used
to grasp visually the quality of recognition but the most interesting command is
`facecame recognize` which can be used to execute bash command on recognition events.

```bash
$ facecam recognize dataset \
        --on "andrea" 'echo "Hello Andrea"'
        --on "unknown" 'echo "go away please"'
```
