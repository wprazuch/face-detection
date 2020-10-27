Face recognition

This app demonstrates how to create a face recognition system based on publicly available deep learning models. The project contains a set of methods and modules,
which enable the user to create a database of people (whose faces in different photos are being represented by an indentifier), which is then used for inference of new
faces. The new faces recognized by the model are being compared with those faces already known for the network, as the comparison with existing keys takes place. If the model
found a face with an identifier similar to the one already in the database, it returns True for a that record. This way, we can recognize faces in the images!

## Data
For face database creation, the [Pins Face Recognition](https://www.kaggle.com/hereisburak/pins-face-recognition/version/1) dataset was used. In consists of face photos
of 105 celebrities.

## Methods

First, a face detection network is used to crop patches of faces from the image. Then, a unique face key is generated on the face cropped from the image. 
The key is then compared with the keys from the database of faces, created before on our faces dataset. If the unknown face's key matches one of the keys from DB,
the person's identification is returned for that matching key. 

Streamlit was created for means of uploading photos to the system for face recognition.

