from pprint import pprint
import time

import cv2

from modules.video_feeder import VideoFeeder
from modules.detector import Detector
from modules.memoryless_violation_evaluator import MemorylessViolationEvaluator
from modules.ui_module import UIModule
from modules.camera_stream_fetcher import CameraStreamFetchersSupervisor
from modules.counter_module import CounterModule

from scripts.frame_visualizer import FrameVisualizerSimple
from scripts.camera import Camera

no_delay_uuids = [
    "22583b92-89c4-4b30-8bb2-a02fa7352e30", #CH14
    "b1fd15ab-fd5e-4b93-a8a7-cde4d3209ac6", #CH15
]

frame_visualizer = FrameVisualizerSimple()
camera_fetchers_supervisor_object = CameraStreamFetchersSupervisor(verbose = True, no_delay_cemara_uuids = no_delay_uuids)
detector_object = Detector(pose_model_index = 0, hard_hat_model_index = 0, forklift_model_index = 0)
memoryless_violation_evaluator_object = MemorylessViolationEvaluator()
counter_module_object = CounterModule()
ui_module_object = UIModule()

camera_fetchers_supervisor_object.start_watching_all_IP_cameras()

# Create a window named 'Object Detection' and set the window to fullscreen if desired
WINDOW_NAME = 'Safety-AI'
IS_FULL_SCREEN = True
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if IS_FULL_SCREEN:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

processed_frame_log = {
 #uuid:fetch_timestamp
}

while True: #process all ip cameras

    recent_frame_details = camera_fetchers_supervisor_object.get_latest_frame_details()
    iteration_detection_results = [] 

    not_processed_frames = []
    for recent_frame_detail in recent_frame_details:
        frame = recent_frame_detail["frame"]
        uuid = recent_frame_detail["uuid"]

        if uuid  in no_delay_uuids: #CH14 and CH15
            not_processed_frames.append(recent_frame_detail)
            continue

        if uuid in processed_frame_log:
            if processed_frame_log[uuid] != recent_frame_detail["fetch_timestamp"]:
                not_processed_frames.append(recent_frame_detail)
                continue
        processed_frame_log[uuid] = recent_frame_detail["fetch_timestamp"]

    for recent_frame_detail in not_processed_frames:
        frame = recent_frame_detail["frame"]
        uuid = recent_frame_detail["uuid"]
        detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_uuid= uuid )
        evaluation_results = memoryless_violation_evaluator_object.evaluate_for_violations(detections = detections, camera_uuid = uuid)
        counter_module_object.update_counters(evaluation_results = evaluation_results)

        evaluation_results["frame_externally_added_key"] = frame        
        iteration_detection_results.append(evaluation_results)

       
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    is_q_pressed, displayed_frame = ui_module_object.update_ui_frame(window_name= WINDOW_NAME, counters = counter_module_object.get_counters(), multiple_camera_evaluation_results=iteration_detection_results, window_scale_factor= 0.75, emoji_scale_factor= 2.25, wait_time_ms= 1, timestamp_str = timestamp_str)
    if is_q_pressed:
        break

# Clean up
cv2.destroyAllWindows()
camera_fetchers_supervisor_object.stop_watching_all_IP_cameras()
