import math,os
import cv2
import time

from ultralytics import YOLO

class ForkliftDetector:
    def __init__(self, model_path:str)-> None:          
        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH )   
        self.recent_prediction_results = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def get_empty_prediction_dict_template(self) -> dict:
        empty_prediction_dict = {   
                    "DETECTOR_TYPE":"ForkliftDetector",          # which detector made this prediction
                    "frame_shape": [0,0],                       # [0,0], [height , width] in pixels
                    "class_name":"",                            # hard_hat, no_hard_hat
                    "bbox_confidence":0,                        # 0.0 to 1.0
                    "bbox_xyxy_px":[0,0,0,0],                        # [x1,y1,x2,y2] in pixels
                    "bbox_center_px": [0,0],                            # [x,y] in pixels
        }
        return empty_prediction_dict

    def predict_frame_and_return_detections(self, frame) -> list[dict]:
        self.recent_prediction_results = []

        results = self.yolo_object(frame, task = "predict", verbose = False)[0]     
        for i, result in enumerate(results):
            boxes = result.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            if box_cls_name not in ["forklift"]:
                continue
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]
            prediction_dict_template = self.get_empty_prediction_dict_template()

            prediction_dict_template["frame_shape"] = list(results.orig_shape)
            prediction_dict_template["class_name"] = box_cls_name
            prediction_dict_template["bbox_confidence"] = box_conf
            prediction_dict_template["bbox_xyxy_px"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            prediction_dict_template["bbox_center_px"] = [ (box_xyxy[0]+box_xyxy[2])/2, (box_xyxy[1]+box_xyxy[3])/2]
            self.recent_prediction_results.append(prediction_dict_template)

        return self.recent_prediction_results