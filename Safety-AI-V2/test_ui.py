from pprint import pprint
import time

import cv2

from modules.video_feeder import VideoFeeder
from modules.detector import Detector
from modules.memoryless_violation_evaluator import MemorylessViolationEvaluator
from modules.ui_module import UIModule

from scripts.frame_visualizer import FrameVisualizerSimple
from scripts.camera import Camera

frame_visualizer = FrameVisualizerSimple()
video_feeder_object = VideoFeeder()
detector_object = Detector(pose_model_index = 0, hard_hat_model_index = 0, forklift_model_index = 0)
memoryless_violation_evaluator_object = MemorylessViolationEvaluator()
ui_module_object = UIModule()

all_recording_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16]

recordings_to_check = all_recording_indexes

# Initialize Video Writer
# Replace 'output_video.mp4' with your desired output file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files, 'XVID' for .avi

video_fps = 3 * 6
video_writer = cv2.VideoWriter('output_video.mp4', fourcc, video_fps, (1440, 810))  # Adjust frame rate and dimensions as needed

skipping_second = 3 #initial skipping second, changes dynamically based on the detection results
while True: #process all recodings
    start_time = time.time()  

    iteration_detection_results = []   
    is_person_detected = False
    for recording_index in recordings_to_check:
        start_time = time.time()
        print(f"Checking recording index: {recording_index}")
        video_feeder_object.change_to_video(recording_index)
        frame, ret, NVR_ip, channel, uuid = video_feeder_object.get_current_video_frame() 
           
        detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_uuid= uuid )
        evaluation_results = memoryless_violation_evaluator_object.evaluate_for_violations(detections = detections, camera_uuid = uuid)
        evaluation_results["frame_externally_added_key"] = frame        
        iteration_detection_results.append(evaluation_results)
        video_feeder_object.fast_forward_seconds(skipping_second)     

        if evaluation_results["number_of_persons"] > 0:
            is_person_detected = True

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"FPS: {1/elapsed_time} Recording index: {recording_index} processed in {elapsed_time} seconds")

    # if not is_person_detected:
    #     skipping_second = min(30, skipping_second + 2.5)
    # else:
    #     skipping_second = 5
        
    timestamp_str = video_feeder_object.get_current_video_date_str()
    is_q_pressed, displayed_frame = ui_module_object.update_ui_frame(multiple_camera_evaluation_results=iteration_detection_results, window_scale_factor= 0.75, emoji_scale_factor= 2.25, wait_time_ms= 1, timestamp_str = timestamp_str)
    if is_q_pressed:
        break     
    video_writer.write(displayed_frame)

    end_time = time.time()
    elapsed_time = end_time - start_time   

    print(f"IPS ={str(round(1/elapsed_time,1)):<5},  FPS = {str(round(len(recordings_to_check)/elapsed_time,1)):<5}, skipping second = {skipping_second}", end="\n") #\r
    

# Release the video writer
video_writer.release()
cv2.destroyAllWindows()
print("Video writer released and windows destroyed")
