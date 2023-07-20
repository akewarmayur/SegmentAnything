from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import subprocess
import glob
from google.colab.patches import cv2_imshow
import random


class SamInference:

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
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        sam.to(device=device)
        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)

        return predictor, mask_generator

    def draw_box_on_image(self, image, all_boxes, color=(0, 255, 0), thickness=2):
        for bbox in all_boxes:
            bbox = tuple(bbox)
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        return image

    def draw_masks_on_image(self, image, masks, colors=None, alpha=0.3):
        if colors is None:
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                      range(len(masks))]

        overlay = image.copy()
        for mask, color in zip(masks, colors):
            overlay[mask] = color

        output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return output

    def draw_circle(self, image, center, thickness=7):
        center = tuple(map(tuple, center))
        size = 5
        color = (0, 0, 0)
        for c in center:
            cv2.circle(image, c, size, color, thickness)
        return image

    def processMask(self, fi, mask_generator, predictor, segment_all, box_all, box_prompt, point_prompt, show_image, colab):
        nameofImage = fi.split("/")
        nameofImage = nameofImage[len(nameofImage) - 1]
        image = cv2.imread(fi)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if segment_all == "True" or segment_all == "TRUE":
            masks = mask_generator.generate(image)
            all_masks = []
            for m in masks:
                all_masks.append(m['segmentation'])
            output_image = self.draw_masks_on_image(image, all_masks)
            if show_image == "True" or show_image == "TRUE":
                if colab == "True" or colab == "TRUE":
                    cv2_imshow(output_image)
                else:
                    cv2.imshow('Image with Masks', output_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                cv2.imwrite("maskedImages/" + nameofImage, output_image)
        elif box_all == "True" or box_all == "TRUE":
            masks = mask_generator.generate(image)
            all_boxes = []
            for m in masks:
                all_boxes.append(m['bbox'])
            output_image = self.draw_box_on_image(image, all_boxes)
            if show_image == "True" or show_image == "TRUE":
                if colab == "True" or colab == "TRUE":
                    cv2_imshow(output_image)
                else:
                    cv2.imshow('Image with Boxes', output_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                cv2.imwrite("boxedImages/" + nameofImage, output_image)

        else:
            if len(box_prompt) != 0 and len(point_prompt) == 0:
                input_box = np.array(list(box_prompt))
                predictor.set_image(image)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                output_image = self.draw_masks_on_image(image, masks)
                if show_image == "True" or show_image == "TRUE":
                    print("Not here")
                    if colab == "True" or colab == "TRUE":
                        cv2_imshow(output_image)
                    else:
                        cv2.imshow('Image with box prompt', output_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print("Should Be Here")
                    cv2.imwrite("promptImages/" + nameofImage, output_image)

            if len(box_prompt) == 0 and len(point_prompt) != 0:
                input_point = np.array([point_prompt])
                input_label = np.array([1 for _ in range(len(input_point))])
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                output_image = self.draw_masks_on_image(image, masks)
                output_image = self.draw_circle(output_image, input_point)
                if show_image == "True" or show_image == "TRUE":
                    if colab == "True" or colab == "TRUE":
                        cv2_imshow(output_image)
                    else:
                        cv2.imshow('Image with point prompt', output_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    cv2.imwrite("promptImages/" + nameofImage, output_image)

            if len(box_prompt) != 0 and len(point_prompt) != 0:
                input_point = np.array([point_prompt])
                input_label = np.array([1 for _ in range(len(point_prompt))])
                input_box = np.array(list(box_prompt))
                predictor.set_image(image)
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=False,
                )
                output_image = self.draw_masks_on_image(image, masks)
                output_image = self.draw_circle(output_image, input_point)
                if show_image == "True" or show_image == "TRUE":
                    if colab == "True" or colab == "TRUE":
                        cv2_imshow(output_image)
                    else:
                        cv2.imshow('Image with point prompt', output_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    cv2.imwrite("promptImages/" + nameofImage, output_image)

    def processImage(self, image_path, segment_all, box_all, box_prompt, point_prompt, show_image, colab):
        isFolder = False
        name = image_path.split(".")
        predictor, mask_generator = self.getModel()
        if len(name) == 1:
            isFolder = True
        if isFolder == "True" or isFolder == "TRUE":
            for fi in glob.glob(image_path + "/*"):
                self.processMask(fi, mask_generator, predictor, segment_all, box_all, box_prompt, point_prompt,
                                 show_image, colab)

        else:
            self.processMask(image_path, mask_generator, predictor, segment_all, box_all, box_prompt, point_prompt,
                             show_image, colab)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--path', action='store', type=str, required=True)
    my_parser.add_argument('--box_prompt', action='store', type=str, required=True)
    my_parser.add_argument('--point_prompt', action='store', type=str, required=True)
    my_parser.add_argument('--segment_all', action='store', type=str, required=True)
    my_parser.add_argument('--box_all', action='store', type=str, required=True)
    my_parser.add_argument('--show_image', action='store', type=str, required=True)
    my_parser.add_argument('--colab', action='store', type=str, required=True)
    args = my_parser.parse_args()
    path = args.path
    segment_all = args.segment_all
    box_all = args.box_all
    box_prompt = eval(args.box_prompt)
    point_prompt = eval(args.point_prompt)
    show_image = args.show_image
    isColab = args.colab
    print(path, segment_all, box_all, box_prompt, point_prompt, show_image, isColab)

    obj = SamInference()
    obj.processImage(path, segment_all, box_all, box_prompt, point_prompt, show_image, isColab)

    # python /content/gdrive/MyDrive/samInference.py --path /content/gdrive/MyDrive/TheIrishmanimg_000001.png
    # --box_prompt "[]" --point_prompt "[]" --segment_all 'True' --box_all 'False' --show_image 'True' --colab 'False'
