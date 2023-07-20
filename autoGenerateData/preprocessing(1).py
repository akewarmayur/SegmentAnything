import os
import subprocess
import re
import glob


class Preprocess:

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def extract_frames(self, framesPath, input_video, UID, fps):
        dir = os.listdir(framesPath)
        if len(dir) == 0:
            print("Empty directory")
            query = "ffmpeg -i " + input_video + " -pix_fmt rgb24 -vf fps=" + str(
                fps) + " " + framesPath + str(UID) + "img_%06d.png"
            response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
            _ = str(response).encode('utf-8')
        frames = []
        for file in glob.glob(framesPath + '/*'):
            frames.append(file)
        frames.sort(key=self.natural_keys)
        return frames

    def createFolders(self, UID):
        dest_folder_path = "extractedFrames/" + str(UID) + "/"
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)
        return dest_folder_path

