import json, mimetypes, os, math
from datetime import datetime, timedelta, timezone
from pprint import pprint

import cv2

class VideoFeeder:

    def __init__(self) -> None:
        #this will automatically initialize the videos to be processed
        self.current_video_index = 0
        self.video_recording_objects = []

        VIDEOS_TO_PROCESS_FOLDER_PATH = os.path.join("videos", "videos_to_process")
        self.VIDEO_PATHS = [os.path.join(VIDEOS_TO_PROCESS_FOLDER_PATH, video) for video in os.listdir(VIDEOS_TO_PROCESS_FOLDER_PATH) if (video.endswith(".mp4") or video.endswith(".avi"))]

        with open('json_files/camera_configs.json') as f:
            camera_configs = json.load(f)["cameras"]
        for video_path in self.VIDEO_PATHS:

            #Example video base name -> NVR-03_XRN-6410RB2_CH014(172.16.0.23)_20240215_080000_085924_ID_0100.avi
            video_base_name  = os.path.basename(video_path)
            splitted_video_base_name = video_base_name.split("_")
            video_channel = splitted_video_base_name[2][0:5]
            vide_NVR_ip = splitted_video_base_name[2][6:17]
            video_YYYYMMDD_str = splitted_video_base_name[3]
            video_start_HHMMSS_str = splitted_video_base_name[4]
            video_end_HHMMSS_str = splitted_video_base_name[5]

            # Create datetime objects for the video start and end times
            tz_offset = timezone(timedelta(hours=3))
            video_start_datetime = datetime.strptime(video_YYYYMMDD_str + video_start_HHMMSS_str, '%Y%m%d%H%M%S').replace(tzinfo=tz_offset)
            video_end_datetime = datetime.strptime(video_YYYYMMDD_str + video_end_HHMMSS_str, '%Y%m%d%H%M%S').replace(tzinfo=tz_offset)

            video_basic_info = {
                "video_path": video_path,
                "video_base_name": os.path.basename(video_path),
                "related_camera_uuid": None,
                "video_channel": video_channel,
                "video_NVR_ip": vide_NVR_ip,
                "video_start_datetime": video_start_datetime,
                "video_end_datetime": video_end_datetime,
            }

            is_camera_config_found = False
            for camera_config in camera_configs:
                if camera_config["channel"] == video_channel and camera_config["NVR_ip"] == vide_NVR_ip:
                    is_camera_config_found = True
                    video_basic_info["related_camera_uuid"] = camera_config["uuid"]
                    break
            if not is_camera_config_found:
                raise ValueError(f"Camera configuration not found for the video: {video_base_name}")
            
            #TODO: create VideoRecording object and attach it to this object
            self.video_recording_objects.append(VideoRecording(video_basic_info))
        
        print(f"VideoFeeder is initialized with {len(self.video_recording_objects)} video(s).")

    def get_current_video_index(self) -> int:
        return self.current_video_index
    
    def get_current_video_datetime(self) -> datetime:
        return self.video_recording_objects[self.current_video_index].get_current_frame_datetime()
    
    def get_current_video_date_str(self) -> str:
        return self.video_recording_objects[self.current_video_index].get_current_frame_datetime().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_current_video_frame(self) -> None:
        frame, ret = self.video_recording_objects[self.current_video_index].get_current_frame()
        NVR_ip = self.video_recording_objects[self.current_video_index].get_NVR_ip()
        channel = self.video_recording_objects[self.current_video_index].get_channel()
        uuid = self.video_recording_objects[self.current_video_index].get_related_camera_uuid()        

        return frame, ret, NVR_ip, channel, uuid
    
    def get_watched_duration_percentage(self) -> float:
        return round(self.video_recording_objects[self.current_video_index].get_watched_duration_percentage(),2)
    
    def change_to_next_video(self) -> bool:
        if self.current_video_index == len(self.video_recording_objects) - 1:
            return False
        self.current_video_index += 1
        return True
    
    def change_to_previous_video(self) -> bool:
        if self.current_video_index == 0:
            return False
        self.current_video_index -= 1
        return True
    
    def change_to_video(self, video_index: int) -> None:
        if video_index < 0 or video_index > len(self.video_recording_objects) - 1:
            raise ValueError("Invalid video index")
        self.current_video_index = video_index
    
    def fast_forward_seconds(self, seconds: int) -> None:
        is_at_edge = self.video_recording_objects[self.current_video_index].fast_forward_seconds(seconds)
        return (not is_at_edge) #if returns False, you should continue to the next video
    
    def fast_backward_seconds(self, seconds: int) -> None:
        is_at_edge = self.video_recording_objects[self.current_video_index].fast_backward_seconds(seconds)
        return (not is_at_edge)
    
    def show_current_frame(self, frame_ratio = 1, close_window_after = False, wait_time_ms:int = 0) -> None:
        self.video_recording_objects[self.current_video_index].show_current_frame(frame_ratio, close_window_after, wait_time_ms)

class VideoRecording:

    def __init__(self, video_basic_info:dict = None):

        self.VIDEO_PATH = video_basic_info["video_path"]
        self.VIDEO_BASE_NAME = video_basic_info["video_base_name"]
        self.RELATED_CAMERA_UUID = video_basic_info["related_camera_uuid"]
        self.VIDEO_CHANNEL = video_basic_info["video_channel"]
        self.VIDEO_NVR_IP = video_basic_info["video_NVR_ip"]
        self.VIDEO_START_DATETIME = video_basic_info["video_start_datetime"]
        self.VIDEO_END_DATETIME = video_basic_info["video_end_datetime"]
        self.TOTAL_SECONDS = (self.VIDEO_END_DATETIME - self.VIDEO_START_DATETIME).total_seconds()

        #mime-type = type "/" [tree "."] subtype ["+" suffix]* [";" parameter];
        #NOTE: this is a regex pattern. Thus may not be the best way to check for the path type. Yet simple one.
        mimetype, _ = mimetypes.guess_type(self.VIDEO_PATH)        
        self.CV2_VIDEO_CAPTURE_OBJECT = cv2.VideoCapture(self.VIDEO_PATH)
        self.MIME_TYPE = mimetype
        self.FRAME_WIDTH = int(self.CV2_VIDEO_CAPTURE_OBJECT.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FRAME_HEIGHT = int(self.CV2_VIDEO_CAPTURE_OBJECT.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.TOTAL_FRAME_COUNT = int(self.CV2_VIDEO_CAPTURE_OBJECT.get(cv2.CAP_PROP_FRAME_COUNT))
        self.VIDEO_FPS = self.CV2_VIDEO_CAPTURE_OBJECT.get(cv2.CAP_PROP_FPS)
        self.VIDEO_FPS_CALCULATED = self.TOTAL_FRAME_COUNT/self.TOTAL_SECONDS #NOTE: this is not the same as the video fps. This is calculated using the total frame count and total seconds derived from video name. This is better to use for calculations.
        self.PERIOD_BETWEEN_FRAMES = 1/self.VIDEO_FPS_CALCULATED

        self.current_frame_index = 0

    def get_channel(self) -> str:
        return self.VIDEO_CHANNEL
    
    def get_NVR_ip(self) -> str:
        return self.VIDEO_NVR_IP
    
    def get_video_basename(self) -> str:
        return self.VIDEO_BASE_NAME
    
    def get_related_camera_uuid(self) -> str:
        return self.RELATED_CAMERA_UUID
    
    def get_watched_duration_percentage(self) -> float:
        return (self.current_frame_index/self.TOTAL_FRAME_COUNT) * 100
    
    def get_current_frame(self) -> None:
        ret, frame = self.CV2_VIDEO_CAPTURE_OBJECT.read()
        #reading increment the frame index by 1, thus we need to ensure that the frame index is set to the correct value.
        self.CV2_VIDEO_CAPTURE_OBJECT.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return frame, ret
           
    def get_current_frame_index(self) -> int:
        return self.current_frame_index
    
    def get_current_frame_second(self) -> float:
        seconds_now = (self.current_frame_index/self.TOTAL_FRAME_COUNT) * self.TOTAL_SECONDS
        return seconds_now
    
    def get_current_frame_datetime(self) -> datetime:
        return self.VIDEO_START_DATETIME + timedelta(seconds = self.get_current_frame_second())

    def set_current_frame_index(self, frame_index: int) -> None:
        if frame_index < 0 or frame_index > (self.TOTAL_FRAME_COUNT-1):
            raise ValueError("Invalid frame index")
        self.current_frame_index = frame_index
        self.CV2_VIDEO_CAPTURE_OBJECT.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
       
    def delta_iterate_current_frame_index(self, delta_frame_index: int) -> bool:
        is_at_edge = True
        if self.current_frame_index+delta_frame_index > (self.TOTAL_FRAME_COUNT-1):
            self.current_frame_index = (self.TOTAL_FRAME_COUNT-1)
        elif self.current_frame_index+delta_frame_index < 0:
            self.current_frame_index = 0
        else:    
            is_at_edge = False            
            self.current_frame_index += delta_frame_index
        
        self.CV2_VIDEO_CAPTURE_OBJECT.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return is_at_edge
    
    def fast_forward_seconds(self, forwarding_seconds: float) -> bool:
        delta_frames = int(max(forwarding_seconds / self.PERIOD_BETWEEN_FRAMES,1))
        is_at_edge = self.delta_iterate_current_frame_index(delta_frames) #means frame is the either first or last frame
        return is_at_edge
    
    def fast_backward_seconds(self, backward_seconds: float) -> bool:
        delta_frames = int(-max(backward_seconds / self.PERIOD_BETWEEN_FRAMES,1))
        is_at_edege = self.delta_iterate_current_frame_index(delta_frames)
        return is_at_edege

    def show_current_frame(self, frame_ratio = 1, close_window_after = False, wait_time_ms:int = 0) -> None:
        # should be only used to test the video, has no purpose in the final product
        if frame_ratio < 0 or frame_ratio > 1:
            raise ValueError("Frame ratio must be between 0 and 1")
        
        frame, ret = self.get_current_frame()
        frame_to_show = frame
        if frame_ratio != 1:  # Only resize if the ratio is not 1
            new_width = int(frame.shape[1] * frame_ratio)
            new_height = int(frame.shape[0] * frame_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            frame_to_show = resized_frame
        
        if ret:
            cv2.imshow("Testing", frame_to_show)
            cv2.waitKey(wait_time_ms)
            if close_window_after:
                cv2.destroyAllWindows()
        
