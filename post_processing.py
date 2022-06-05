from scipy.spatial.distance import pdist, squareform
from torchvision.models import resnext101_32x8d
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import Identity
from scipy.stats import skew
from glob import glob
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import numpy as np
import colorsys
import json
import cv2
import os

BLUE = colorsys.rgb_to_hsv(0, 0, 255)
GREEN = colorsys.rgb_to_hsv(0, 255, 0)
V_DIFF = GREEN[0] - BLUE[0]

FONT = cv2.FONT_HERSHEY_SIMPLEX


class PostProcessor:
    def __init__(self, folder_dir):
        self.folder_dir = folder_dir

        frame_diffs_file = os.path.join(self.folder_dir, "frame_diffs.json")
        with open(frame_diffs_file) as f:
            self.frame_diffs = json.load(f)

        vid_info_file = os.path.join(self.folder_dir, "vid_info.json")
        with open(vid_info_file) as f:
            self.vid_info = json.load(f)

        self.obj_folders = glob(os.path.join(self.folder_dir, "*/object_*"))
        
        self.backbone = Backbone()

    def run(self):
        for folder in tqdm(self.obj_folders):
            self._generate_embeddings(folder)
            self._generate_heatmap(folder)
        self._generate_obj_relations()

    def _generate_embeddings(self, folder):
        images = glob(os.path.join(folder, "*.jpg"))

        embeds = []
        for img_path in images:
            img = Image.open(img_path)
            emb = self.backbone(img)
            embeds.append(emb.detach().numpy())

        embeds = np.concatenate(embeds, axis=0)
        np.save(os.path.join(folder, "embeddings.npy"), embeds)

    def _generate_heatmap(self, folder):
        width = self.vid_info["width"]
        height = self.vid_info["height"]

        background = np.zeros((height, width, 3), np.uint8)

        with open(os.path.join(folder, "info.json")) as f:
            info = json.load(f)

        total_frames = info["total_frames"]
        total_duration = total_frames/self.vid_info["fps"]

        dists = np.array(list(info["distance_to_prev_frame"] for info in info["frames"]))

        anomalies = self.get_anomalis(dists)

        prev_center = None
        for i, frame_info in enumerate(info["frames"]):
            color = (BLUE[0]+i*V_DIFF/total_frames, BLUE[1], BLUE[2])
            color = list(map(int, colorsys.hsv_to_rgb(*color)))[::-1]
            cv2.circle(background, frame_info["center"], 3, color, -1)
            if prev_center is not None:
                if i in anomalies:
                    color = (0, 0, 255)
                cv2.line(background, prev_center, frame_info["center"], (*color, 0.1), 1)

            prev_center = frame_info["center"]


        px = np.array((0, 15))
        pos = np.array((20, 20))

        txt = f"Numer of Frames: {total_frames}"
        cv2.putText(background, txt, pos, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        start_time = self.s2t(info["first_frame"]/self.vid_info["fps"])
        end_time = self.s2t(info["last_frame"]/self.vid_info["fps"])
        txt = f"Appearance duration: {total_duration:.2f} from {start_time} to {end_time}"
        cv2.putText(background, txt, pos+px, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        overal_dist, rel_mean, rel_skew, fluctuation = self._get_direction(info) 
        txt = f"Overal object moved to {['right', 'left'][overal_dist < 0]} for {abs(overal_dist)} pixels on screen."
        cv2.putText(background, txt, pos+px*2, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        txt = f"Object on average moved to {['right', 'left'][int(rel_mean < 0)]} for {abs(rel_mean):.2f} pixels on screen each frame."
        cv2.putText(background, txt, pos+px*3, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        txt = f"Object mostly moved to {['right', 'left'][rel_skew < 0]}."
        cv2.putText(background, txt, pos+px*4, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        txt = f"Object fluctuates in its direction {int(fluctuation*100):02}% of times."
        cv2.putText(background, txt, pos+px*5, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        heat_map_path = os.path.join(folder, "heatmap.jpg")
        # cv2.imshow("ASd", background)
        # cv2.waitKey(0)
        cv2.imwrite(heat_map_path, background)

    def _generate_obj_relations(self):

        embedding_file_paths = glob(os.path.join(self.folder_dir, "*/*/*.npy"))
        embedding_names = ["__".join(path.split(os.sep)[-3:-1]) for path in embedding_file_paths]
        embeddings = np.stack([np.load(path).mean(axis=0) for path in embedding_file_paths])
        dist_matrix = squareform(pdist(embeddings))

        plt.figure(figsize=(10, 10))
        cmap = sns.color_palette("dark:salmon_r", as_cmap=True)
        plot = sns.heatmap(dist_matrix, xticklabels=embedding_names, yticklabels=embedding_names,
                             vmin=0, vmax=0.55, cmap=cmap, annot=True)

        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right")
        plot.set_yticklabels(plot.get_yticklabels(), rotation=45, ha="right")
        plt.title("Embedding distance of all objects")
        plot.get_figure().savefig("test.png")

    def _generate_jump_shot_timeline(self):
        diffs = self.frame_diffs

        

    @staticmethod
    def _get_direction(info):
        first_center = info["first_center"]
        last_center = info["last_center"]

        overal_dist = last_center[0] - first_center[0]

        rel_dists = np.array(list(data["rel_dist_to_prev_frame"] for data in info["frames"]))

        rel_mean = rel_dists.mean()
        rel_skew = skew(rel_dists)

        fluctuation = (rel_dists > 0).mean()

        return overal_dist, rel_mean, rel_skew, fluctuation

    @staticmethod
    def s2t(sec):
        m, s = divmod(sec, 60)
        return f"{int(m)}:{int(s):02}"

    @staticmethod
    def get_anomalis(data, sigma=3):
        anomalies = []
        
        data_std = data.std()
        data_mean = data.mean()
        anomaly_cut_off = data_std * sigma
        upper_limit = data_mean + anomaly_cut_off
        anomaly_indices = np.where(data > upper_limit)[0]

        return anomaly_indices


class Backbone:
    def __init__(self):
        self.model = resnext101_32x8d(pretrained=True)
        self.model.fc = Identity()
        self.transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    def __call__(self, img):
        return self.model(self.transforms(img).unsqueeze(0))


PostProcessor("results/20220605__220131_video_1").run()