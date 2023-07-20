import cv2
import torch
import clip
from PIL import Image
from itertools import islice
import sys
import os
import argparse
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import subprocess
import glob
import class_map_config


class generateData:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def draw_bounding_box(self, image, bbox, color=(0, 255, 0), thickness=2):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    def download_model(self, url, save_path):
        command = f'wget -O {save_path} {url}'
        subprocess.call(command, shell=True)

    def get_sam_model(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        dir = os.listdir("models")
        if len(dir) == 0:
            self.download_model("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                                "models/sam_vit_h_4b8939.pth")

    def get_clip_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def get_prediction(self, frame_path, list_of_labels, how_many_predictions, model, preprocess) -> list:
        Highest3Predictions = []
        try:
            text = clip.tokenize(list_of_labels).to(self.device)
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs = probs.tolist()[0]
            vv = {}
            for i, j in enumerate(probs):
                vv[list_of_labels[i]] = j
            maxx = {k: v for k, v in sorted(vv.items(), key=lambda item: item[1], reverse=True)}
            Highest3Predictions = list(islice(maxx.items(), how_many_predictions))
            print(f"{frame_path} : {Highest3Predictions}")
        except:
            pass

        return Highest3Predictions

    def initialize_models(self):
        model, preprocess = self.get_clip_model()
        self.get_sam_model()
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator, model, preprocess

    def calculate_normalized_coordinates(self, x1, y1, x2, y2, img):
        image_height, image_width = img.shape[:2]
        x_min_normalized = x1 / image_width
        y_min_normalized = y1 / image_height
        x_max_normalized = x2 / image_width
        y_max_normalized = y2 / image_height
        return x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized

    def getBox(self, SOURCE_IMAGE_PATH, name, extension, OBJECTS, list_of_prompts, mask_generator, model, preprocess, CLIP_THRESHOLD):
        print(SOURCE_IMAGE_PATH)
        how_many_predictions = 2
        color = (0, 255, 0)  # Green color (BGR format)
        thickness = 2  # Thickness of the bounding box
        destination = "boxedImages/"
        if not os.path.exists("croppedBoxes"):
            os.makedirs("croppedBoxes")
        if not os.path.exists(destination):
            os.makedirs(destination)
        try:
            image = cv2.imread(SOURCE_IMAGE_PATH)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            for aa, m in enumerate(masks):
                bbox = tuple(m['bbox'])
                # x1, y1, x2, y2 = bbox
                x, y, width, height = bbox
                if height >= 50 and width >= 50:
                    cropped_image = image[y:y + height, x:x + width]
                    # cropped_image = image[y1:y2, x1:x2]
                    cv2.imwrite("croppedBoxes/" + str(aa) + ".jpg", cropped_image)
                    Highest3Predictions = self.get_prediction("croppedBoxes/" + str(aa) + ".jpg", list_of_prompts,
                                                              how_many_predictions, model, preprocess)
                    c1 = Highest3Predictions[0][0]
                    s1 = round(100 * Highest3Predictions[0][1], 2)
                    if c1 == "a photo of others":
                        pass
                    elif c1 == "a photo ":
                        pass
                    elif c1 == "a photo of unrecognized object":
                        pass
                    else:
                        bboxes = []
                        if s1 >= CLIP_THRESHOLD:
                            bboxes.append([x, y, width, height])
                            with open("yoloV8Files/" + str(name) + '.txt', 'w') as f:
                                for a, b in enumerate(bboxes):
                                    print(b)
                                    x1, y1, x2, y2 = b
                                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                                    class_name = OBJECTS[a]
                                    class_code = class_map_config.class_map[class_name]
                                    x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized = self.calculate_normalized_coordinates(
                                        x1, y1, x2, y2, image)
                                    f.write(str(class_code) + ' ' + str(x_min_normalized) + ' ' + str(
                                        y_min_normalized) + ' ' + str(
                                        x_max_normalized) + ' ' + str(y_max_normalized))
                                    f.write("\n")
                            cv2.imwrite(destination + "boxedImages/" + name + "." + extension, image)

        except Exception as e:
            print(e)

    def startProcess(self, folder_name, OBJECTS, CLIP_THRESHOLD):
        list_of_prompts = []
        for o in OBJECTS:
            list_of_prompts.append("a photo of " + str(o))
        list_of_prompts.append("a photo of others")
        list_of_prompts.append("a photo ")
        list_of_prompts.append("a photo of unrecognized object")
        mask_generator, model, preprocess = self.initialize_models()
        for fi in glob.glob(folder_name + "/*"):
            try:
                SOURCE_IMAGE_PATH = fi
                name = SOURCE_IMAGE_PATH.split("/")
                extension = name[len(name) - 1].split(".")[1]
                name = name[len(name) - 1].split(".")[0]
                self.getBox(SOURCE_IMAGE_PATH, name, extension, OBJECTS, list_of_prompts, mask_generator, model,
                            preprocess, CLIP_THRESHOLD)
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
    CLIP_THRESHOLD = 95
    videos = False
    objT = generateData()
    objT.startProcess(folder_name, OBJECTS, CLIP_THRESHOLD)
