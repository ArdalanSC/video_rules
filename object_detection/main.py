import torch
import numpy as np
import cv2
from .utils.general import check_img_size, non_max_suppression, scale_coords, validate_cordinates
from .utils.augmentations import  letterbox
from .models.yolo import Model
import PIL
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class YoloModel:
    def __init__(self, model_config: str,  weights: str, imgsz: int = 640,
                 conf_thres: float = 0.55, iou_thres: float = 0.45, device: str = 'cpu'):
        """
        Args:
            model_config ([str]): [path to the model's config file]
            weights ([str]): [path to the weights file]
            imgsz (int, optional): [size of the input image to the detection model]. Defaults to 640.
            conf_thres (float, optional): [detections with less confidence that the value will be ignored]. Defaults to 0.75.
            iou_thres (float, optional): [NMS IoU threshold]. Defaults to 0.45.
            device (str, optional): [weather do the evaluation on cpu or gpu]. Defaults to 'cuda:0'.
        """
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_config = os.path.join(self.DIR, model_config)
        self.path = os.path.join(self.DIR, weights)
        self.device = device
        self.model = self._load_model(self.model_config, self.path, self.device)
        self.image_size = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

    @staticmethod
    def _load_model(config: str, path: str, device: str) -> torch.nn.Module:
        """[takes the path and the device and loads the model on the preferred device read for evaluation]

        Args:
            config ([str]): [path to the model's config file]
            path ([str]): [path to the model's weights file]
            device ([str]): [weather to load the model on cpu or gpu]

        Returns:
            [torch.Model]: [loaded model]
        """
        model = Model(cfg=config)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'].state_dict())
        model = model.to(device)
        model = model.float().fuse()
        model.eval()
        return model

    def _img_preprocess(self, image: (np.ndarray, PIL.Image.Image)) -> torch.Tensor:
        """[takes image array and applies transformation]

        Args:
            image ([np.ndarray]): [input image to transform]

        Returns:
            [torch.Tensor]: [transformed image tensor]
        """

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        imgsz = check_img_size(self.image_size, s=self.model.stride.max())
        img = letterbox(image, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def __call__(self, image: PIL.Image.Image) -> dict:
        """[takes input PIL image, evaluates and returns detection results]

        Args:
            image ([PIL.Image.Image]): [input image to evaluate]

        Returns:
            [dict]: [dictionary of labels, coordinates] if there is a detection
            and empty list otherwise
        """

        results = []
        if not isinstance(image, PIL.Image.Image):
            return results
        if image.mode != "RGB":
            image = image.convert('RGB')
        w, h = image.size
        img = self._img_preprocess(image)
        with torch.no_grad():
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], (h, w)).round()
                for d in det:

                    label = self.names[int(d[5])]
                    if label != "person":
                        continue
                    cordinate = d[:4].tolist()
                    if not validate_cordinates(cordinate, (h, w)):
                        continue
                    detections = {'label': label,
                                  'coordinate': cordinate}
                    results.append(detections)
        return results
