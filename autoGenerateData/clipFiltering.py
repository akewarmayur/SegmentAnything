import pandas as pd
import shutil
import torch
import clip
from PIL import Image
from itertools import islice


class ClipFiltering:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def get_batch_prediction(self, frame_path_list, list_of_labels, how_many_predictions, model, preprocess):
        predictions = {}
        for fi in frame_path_list:
            Highest3Predictions = self.get_prediction(fi, list_of_labels, how_many_predictions, model, preprocess)
            if len(Highest3Predictions) != 0:
                predictions[fi] = Highest3Predictions
        return predictions

    def startFiltering(self, list_of_frames, list_of_objects, model, preprocess):
        list_of_prompts = []
        for o in list_of_objects:
            list_of_prompts.append("a photo of " + str(o))
        list_of_prompts.append("a photo of others")
        list_of_prompts.append("a photo ")
        list_of_prompts.append("a photo of unrecognized object")

        dest = "clipFilteredData/"
        list_of_labels_first = list_of_prompts
        list_of_images = list_of_frames
        how_many_predictions = 3
        batch_predictions = self.get_batch_prediction(list_of_images, list_of_labels_first,
                                                               how_many_predictions, model, preprocess)
        for key, Highest3Predictions in batch_predictions.items():
            name = key.split("/")
            name = name[len(name) - 1]
            print("***********", name)
            c1 = Highest3Predictions[0][0]
            s1 = round(100 * Highest3Predictions[0][1], 2)
            # c2 = Highest3Predictions[1][0]
            # s2 = round(100 * Highest3Predictions[1][1], 2)
            # c3 = Highest3Predictions[2][0]
            # s3 = round(100 * Highest3Predictions[2][1], 2)
            if c1 == "a photo of others":
                pass
            elif c1 == "a photo ":
                pass
            elif c1 == "a photo of unrecognized object":
                pass
            else:
                if s1 >= 50:
                    shutil.copy(key, dest + name)
        return True
