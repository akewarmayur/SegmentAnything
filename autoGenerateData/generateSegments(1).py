import clip
import pandas as pd
import glob
from preprocessing import Preprocess
from clipFiltering import ClipFiltering
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import cv2
import numpy as np
from segment_anything import SamPredictor
import math
import glob
import torch
import os
import argparse
from groundingdino.util.inference import Model
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
            self.download_model("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                                "models/sam_vit_h_4b8939.pth")
            self.download_model("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
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

    def get_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def normalize_coordinates(self, row_i, col_j, img):
        num_rows, num_cols = img.shape[:2]
        x = col_j / (num_cols - 1.)
        y = row_i / (num_rows - 1.)
        return x, y

    def getMask(self, grounding_dino_model, sam_predictor, SOURCE_IMAGE_PATH, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD,
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
            print("_________________")
            print(detections)
            print("__________________")
            detections.mask = self.segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            class_map = class_map_config.class_map
            maskSS = detections.mask
            labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
            print(labels)
            if len(labels) == 0:
                try:
                    os.remove(SOURCE_IMAGE_PATH)
                except:
                    pass
            else:
                try:
                    cll = [i.split(" ")[0] for i in labels]
                    print(cll)
                    shutil.copy(SOURCE_IMAGE_PATH, destination + name + "." + extension)
                    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
                    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections,
                                                             labels=labels)
                    with open(destination + "yoloV8Files/" + str(name) + '.txt', 'w') as f:
                        for qq, x in enumerate(maskSS):
                            w = class_map[cll[qq]]
                            im = image
                            im[:, :] = [0, 0, 0]
                            pixelpoints = np.array(np.nonzero(x))
                            pixelpoints = np.array([pixelpoints[1], pixelpoints[0]]).T
                            pts = pixelpoints.reshape((-1, 1, 2))
                            cv2.polylines(im, pts, True, (255, 255, 255), 2)
                            # cv2.polylines(annotated_image, pts, True, (255, 255, 255), 2)
                            ret, threshed_img = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 10, 200,
                                                              cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # Convert the contour to the format required for segmentation in COCO format
                            # Write bounding boxes to file in YOLO format
                            for contour in contours:
                                segmentation = contour.flatten().tolist()
                                # x, y, w, h = cv2.boundingRect(contour)
                                # Convert the coordinates to YOLO format and write to file
                                partitioned_list = [segmentation[i:i + 2] for i in range(0, len(segmentation), 2)]
                                f.write(str(w) + ' ')
                                for x, y in partitioned_list:
                                    x, y = self.normalize_coordinates(y, x, image)
                                    f.write(str(x) + ' ' + str(y) + ' ')
                                f.write("\n")
                                break
                    cv2.imwrite(destination + "maskedImages/" + name + "." + extension, annotated_image)
                except Exception as e:
                    print(e)
                    try:
                        os.remove(destination + "yoloV8Files/" + str(name) + '.txt')
                    except:
                        pass
        except Exception as e:
            print(e)
            try:
                os.remove(destination + SOURCE_IMAGE_PATH)
            except:
                pass

            try:
                os.remove(destination + "yoloV8Files/" + str(name) + '.txt')
            except:
                pass

    def removeFiles(self, folder_path):
        removing_files = glob.glob(folder_path + '/*')
        for i in removing_files:
            try:
                os.remove(i)
            except:
                pass

    def startProcessVideo(self, UID, input_video_path, list_of_objects, fps):
        print("Started")
        objPreprocess = Preprocess()
        list_of_frames = objPreprocess.extract_frames("extractedFrames", input_video_path, UID, fps)
        model, preprocess = self.get_model()
        objClip = ClipFiltering()
        st = objClip.startFiltering(list_of_frames, list_of_objects, model, preprocess)

    def startProcess(self, folder_name, CLASSES, BOX_TRESHOLD, TEXT_TRESHOLD):
        GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = "models/groundingdino_swint_ogc.pth"
        SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"
        SAM_ENCODER_VERSION = "vit_h"
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
        sam_predictor = SamPredictor(sam)

        for fi in glob.glob(folder_name + "/*"):
            try:
                SOURCE_IMAGE_PATH = fi
                name = SOURCE_IMAGE_PATH.split("/")
                extension = name[len(name) - 1].split(".")[1]
                name = name[len(name) - 1].split(".")[0]
                self.getMask(grounding_dino_model, sam_predictor, SOURCE_IMAGE_PATH, CLASSES, BOX_TRESHOLD,
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
