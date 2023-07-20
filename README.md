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
### Model Path
```https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth```

### SAM Inference
```
python SAM/samInference.py --path 'image_path or folder_containing_images' --box_prompt "[]" --point_prompt "[]" --segment_all 'True' --box_all 'False' --show_image 'True' --colab 'False'
```
* --path : folder or image path
* --box_prompt : provide list of boxes in (x1,y1,x2,y2) format #[40,50,100,300]
* --point_prompt : provide list of points in (x,y) format # [20,30]
* --segment_all : segment entire image or not
* --box_all : box entire image or not
* --show_image : show image while executing
* --colab : running on colab or not
  
The output will be stored in boxedImages, maskedImages and promptImages folder based on the options
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/2e4fdcfd-a7a7-4e35-9704-f605defa2c59)

![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/dd6d6f1b-b79f-45ff-834e-9ff85d403aa1)

### Generate Data
This uses [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) model with SAM model
Grounding DINO model provide bounding box as per input class
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/c1a04d30-ef34-4d0a-8fc9-51a6b6fff56b)

It will generate segmented or boxed data and its corresponding yolo v8 label file

#### Instance Segmentation Data
```
python autoGenerateData/generateSegments.py --folder_name name_of_folder_containing_images --object_names truck,wheels
```
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/44d410f2-2e5c-4b75-a213-3cd4370cb84a)

#### Object Detection Data
```
python autoGenerateData/generateBoxes.py --folder_name name_of_folder_containing_images --object_names truck,wheels
```
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/f8fdea10-88c8-4f18-8b3a-8a3d44001266)

### CLIP SAM
This uses the capebility of [CLIP](https://github.com/openai/CLIP) model to classify images by inputing prompt

#### Generate Data using CLIP & SAM
```
python clipSAMgenerateData.py --folder_name name_of_folder_containing_images --object_names car,wheels
```

#### Detect Objects using CLIP & SAM
```
python detectObjects.py --folder_name name_of_folder_containing_images --object_names truck,glass
```
Results : [('a photo of a car', 0.955078125), ('a photo of a glass', 0.017486572265625)]
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/90b61362-5a5b-4e13-b64c-ce1b046eb50a)

#### Classify Images using CLIP & SAM
```
python clipSAM/classifyImage.py --folder_name name_of_folder_containing_images --object_names car,wheels
```
![image](https://github.com/akewarmayur/SegmentAnything/assets/31464781/bb82ee70-f44b-459f-a890-49e2bdac29bc)

### Notebooks
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/GroundedSegmentAnything.ipynb) for processing image using SAM and Grounding DINO
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/GroundingDINO.ipynb) for processing image using Grounding DINO
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/SAMCLIP.ipynb) for processing image using CLIP and SAM
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/SAMProcess.ipynb) for processing using SAM model
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/SegmentAnythingGradioApp.ipynb) Gradio App
* [Notebook](https://github.com/akewarmayur/SegmentAnything/blob/master/notebooks/fastSAM.ipynb) for fastSAM







