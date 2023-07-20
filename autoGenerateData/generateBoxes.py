import clip
import pandas as pd
import glob
from preprocessing import Preprocess
from clipFiltering import ClipFiltering
from typing import List
import cv2
import numpy as np
import math
import glob
import torch
import os
import argparse
from groundingdino.util.inference import Model, annotate
import shutil
import supervision as sv
import class_map_config
import subprocess


class generateData:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def download_model(self, url, save_path):
        command = f'wget -O {save_path} {url}'
        subprocess.call(command, shell=True)

    def getModel(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        dir = os.listdir("models")
        if len(dir) == 0:
            self.download_model(
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
                "models/groundingdino_swint_ogc.pth")

    def formatFeatureKeyDF(self, x):
        try:
            aa = []
            ss = str(x).split(",")
            for i in ss:
                i = i.replace("[", "").replace("]", "").strip()
                aa.append(str(round((float(i) * 100), 2)))

            return ",".join(aa)
        except:
            return x

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]

    def calculate_normalized_coordinates(self, x1, y1, x2, y2, img):
        image_height, image_width = img.shape[:2]
        x_min_normalized = x1 / image_width
        y_min_normalized = y1 / image_height
        x_max_normalized = x2 / image_width
        y_max_normalized = y2 / image_height
        return x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized

    def startProcessVideo(self, UID, input_video_path, list_of_objects, fps):
        print("Started")
        objPreprocess = Preprocess()
        list_of_frames = objPreprocess.extract_frames("extractedFrames", input_video_path, UID, fps)
        model, preprocess = self.get_model()
        objClip = ClipFiltering()
        st = objClip.startFiltering(list_of_frames, list_of_objects, model, preprocess)

    def getBox(self, grounding_dino_model, SOURCE_IMAGE_PATH, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD,
                name, extension):
        print(SOURCE_IMAGE_PATH)
        destination = "maskedData/"
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        try:
            image = cv2.imread(SOURCE_IMAGE_PATH)
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=self.enhance_class_name(class_names=CLASSES),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD)
            # convert detections to masks
            bboxes = detections.xyxy.tolist()
            # classes = detections.class_id.tolist()
            color = (0, 255, 0)  # Green color (BGR format)
            thickness = 2  # Thickness of the bounding box
            if len(bboxes) != 0:
                with open("yoloV8Files/" + str(name) + '.txt', 'w') as f:
                    for a, b in enumerate(bboxes):
                        print(b)
                        x1, y1, x2, y2 = b
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                        class_name = CLASSES[a]
                        class_code = class_map_config.class_map[class_name]
                        x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized = self.calculate_normalized_coordinates(
                            x1, y1, x2, y2, image)
                        f.write(str(class_code) + ' ' + str(x_min_normalized) + ' ' + str(y_min_normalized) + ' ' + str(
                            x_max_normalized) + ' ' + str(y_max_normalized))
                        f.write("\n")
                cv2.imwrite(destination + "boxedImages/" + name + "." + extension, image)
        except Exception as e:
            print(e)
            try:
                os.remove(destination + SOURCE_IMAGE_PATH)
            except:
                pass

    def removeFiles(self, folder_path):
        removing_files = glob.glob(folder_path + '/*')
        for i in removing_files:
            try:
                os.remove(i)
            except:
                pass

    def startProcess(self, folder_name, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD):
        GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = "models/groundingdino_swint_ogc.pth"
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        for fi in glob.glob(folder_name + "/*"):
            try:
                SOURCE_IMAGE_PATH = fi
                name = SOURCE_IMAGE_PATH.split("/")
                extension = name[len(name) - 1].split(".")[1]
                name = name[len(name) - 1].split(".")[0]
                self.getBox(grounding_dino_model, SOURCE_IMAGE_PATH, CLASSES, BOX_TRESHOLD,
                             TEXT_TRESHOLD, name, extension)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--folder_name', action='store', type=str, required=True)
    my_parser.add_argument('--object_names', action='store', type=str, required=True)
    args = my_parser.parse_args()
    folder_name = args.folder_name
    bbox = args.bbox
    OBJECTS = args.object_names.split(",")
    videos = False
    objT = generateData()
    for fname in os.listdir(folder_name):
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg'):
            break
    else:
        videos = True
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.25
    if videos:
        for fi in glob.glob(folder_name + "/*"):
            SOURCE_VIDEO_PATH = fi
            name = SOURCE_VIDEO_PATH.split("/")
            name = name[len(name) - 1].split(".")[0]
            objT.startProcessVideo(name, SOURCE_VIDEO_PATH, OBJECTS, 1)
        objT.startProcess("clipFilteredData", OBJECTS, BOX_TRESHOLD, TEXT_TRESHOLD)
    else:
        objT.startProcess(folder_name, OBJECTS, BOX_TRESHOLD, TEXT_TRESHOLD)
