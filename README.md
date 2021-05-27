# emotion-recognition

This repository contains working code used to train an emotion recognition model that can be used to automatically assess audience experience. The training code can be found in `Emotion_Recognition_Module.ipynb`. We also provide an implementation of the full experience assessment pipeline, which captures video from a webcam, identifies and tracks the people in the video, and assigns experience scores for each person found.

# Dependencies

The training script requires the following dependencies

- `CV2`: 4.5.2
- `PIL`: 8.2.0
- `torch`: 1.8.1+cu111
- `torchvision`: 0.9.1+cu111
- `matplotlib`: 3.4.2
- `sklearn`: 0.24.2
- `numpy`: 1.20.3

In addition, the full pipeline makes use of darknet to perform detection using YOLOv4
Note that although we used CUDA, this is not a mandatory requirement.

# How to use

After installing the dependencies, move the \*.py files into the darknet directory and run `python full_pipeline.py`, or launch a jupyter server if you want to train the system.
