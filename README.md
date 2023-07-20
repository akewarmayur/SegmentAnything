# Segment Anything
### This repository contains the code for SAM infernce and the various applications of the Segment Anything Model
* The application includes data generation for training YOLO V8 object detection model
* Classify images using SAM and CLIP model
* Detect objects using SAM and CLIP model
* Detect objects using Grounding DINO model
* Detect objects using Grounding DINO and SAM model

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/757bbda2-9543-44b7-bf6a-15dfd164899b)

## Installation
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

