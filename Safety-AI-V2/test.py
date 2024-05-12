from pprint import pprint
import time

import cv2

from modules.video_feeder import VideoFeeder
from modules.detector import Detector
from modules.memoryless_violation_evaluator import MemorylessViolationEvaluator

from scripts.frame_visualizer import FrameVisualizerSimple
from scripts.camera import Camera

frame_visualizer = FrameVisualizerSimple()
video_feeder_object = VideoFeeder()
detector_object = Detector(pose_model_index = 4, hard_hat_model_index = 0, forklift_model_index = 0)
memoryless_violation_evaluator_object = MemorylessViolationEvaluator()

video_feeder_object.change_to_video(2)
skipping_second = 60

while True:
    start_time = time.time()    
    frame, ret, NVR_ip, channel, uuid = video_feeder_object.get_current_video_frame()    
    if not ret:
        cv2.destroyAllWindows()  
        if not video_feeder_object.change_to_next_video():                    
            break
        print()
        continue
    
    detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_uuid= uuid )
    evaluation_results = memoryless_violation_evaluator_object.evaluate_for_violations(detections = detections, camera_uuid = uuid)
    pprint(evaluation_results)

    if not frame_visualizer.show_frame(frame_name=f"{NVR_ip} - {channel}", frame = frame, detections = detections, scale_factor= 0.75, wait_time_ms= 1):
        break

    video_feeder_object.fast_forward_seconds(skipping_second)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{channel} - % {video_feeder_object.get_watched_duration_percentage()} complete FPS = {str(round(1/elapsed_time,1)):<5}, {uuid}", end="\n") #\r


