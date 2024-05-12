import cv2
import numpy as np

class FrameVisualizerSimple:
    def __init__(self):     
        pass       

    def show_frame(self, frame_name:str="FrameVisualizer", frame = None, detections: dict=None, scale_factor:float = 1, wait_time_ms:int = 0)-> bool:
               
        #draw forklifts
        for prediction in detections["forklift_detections"]:
            if prediction["DETECTOR_TYPE"] == "ForkliftDetector":
                self.__draw_forklift(frame= frame, prediction= prediction)

        # #draw pose detections
        for prediction in detections["pose_detections"]:
            if prediction["DETECTOR_TYPE"] == "PoseDetector":
                self.__draw_person(frame= frame, prediction= prediction)

        #draw hard-hat detections
        for prediction in detections["hard_hat_detections"]:
            if prediction["DETECTOR_TYPE"] == "HardHatDetector":
                self.__draw_hard_hat(frame= frame, prediction= prediction)
            
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow(frame_name, resized_frame)

        key = cv2.waitKey(wait_time_ms)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

    def __draw_forklift(self, frame, prediction:list=None):
        class_name = prediction["class_name"]
        #map() in Python 3 returns a map object (a lazy iterator), so we use list casting for immediate evaluation.
        bbox_xyxy = list(map(int, prediction["bbox_xyxy_px"]))     
        bbox_center = list(map(int, prediction["bbox_center_px"]))
        bbox_confidence = prediction["bbox_confidence"]

        color = (0, 255, 0) if bbox_confidence >0.50 else (0, 75, 0)

        frame = cv2.rectangle(frame, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), color, 2)
        frame = cv2.putText(frame, class_name, (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        frame = cv2.circle(frame, (int(bbox_center[0]), int(bbox_center[1])), 5, (0, 0, 255), -1)

    def __draw_person(self, frame, prediction:list = None):
        class_name = prediction["class_name"]
        bbox_xyxy = list(map(int, prediction["bbox_xyxy_px"]))     
        bbox_center = list(map(int, prediction["bbox_center_px"]))
        bbox_confidence = prediction["bbox_confidence"]

        color = (0, 255, 0) if bbox_confidence >0.50 else (0, 75, 0)
        
        frame = cv2.rectangle(frame, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), color, 2)
        frame = cv2.putText(frame, class_name, (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        frame = cv2.circle(frame, (int(bbox_center[0]), int(bbox_center[1])), 5, (0, 0, 255), -1)

    def __draw_hard_hat(self, frame, prediction:list = None):
        class_name = prediction["class_name"]
        bbox_xyxy = list(map(int, prediction["bbox_xyxy_px"]))     
        bbox_center = list(map(int, prediction["bbox_center_px"]))
        bbox_confidence = prediction["bbox_confidence"]

        color = (0, 255, 0) if class_name == "hard_hat" else (0, 0, 255)
        color = (0, 0, 0) if bbox_confidence < 0.75 else color

        frame = cv2.rectangle(frame, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), color, 2)
        frame = cv2.putText(frame, class_name, (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        frame = cv2.circle(frame, (int(bbox_center[0]), int(bbox_center[1])), 5, (255, 255, 255), -1)