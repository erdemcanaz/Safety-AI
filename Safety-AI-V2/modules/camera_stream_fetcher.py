import pprint, random, time, threading
import cv2  
import json
import numpy as np

class CameraStreamFetchersSupervisor:    
    def __init__(self, verbose:bool = False, no_delay_cemara_uuids:list[str] = []):
        self.camera_configs_list = []
        self.initilize_camera_configs_list()

        self.VERBOSE = verbose
        self.camera_stream_fetchers = []
        self.initilize_camera_stream_fetchers(no_delay_cemara_uuids)

        if self.VERBOSE: print("CameraStreamFetchersSupervisor object created")

    def initilize_camera_configs_list(self) -> None:
        with open('json_files/camera_configs.json') as f:            
            camera_configs_list = json.load(f)["cameras"]          
        self.camera_configs_list = camera_configs_list

    def initilize_camera_stream_fetchers(self, no_delay_cemara_uuids:list[str]=[]) -> None:
        self.uuids_to_consider = [
            "94edc97e-1c91-49da-8004-f4a1b7ef1360",
            "22583b92-89c4-4b30-8bb2-a02fa7352e30",
            "b1fd15ab-fd5e-4b93-a8a7-cde4d3209ac6",
            "00d853cb-4e29-4289-8cd9-13918fefb9e7",
            "306f2a75-6240-4ae3-a930-a9b895cdcba7",
            "47968b23-1fd5-4a96-827f-9bc1b94cfd74",
            "be06a307-c9e9-41da-a411-fa5b365e4a07",
            "94b80fe6-5fd9-481a-8cbb-00be0e1dfb1c",
            "7cabf973-f717-44a7-a261-2a3ec7cc610c",
            "92780f91-d255-41a9-acec-65af3070a7bc",           
            "d1a09fa8-80fc-4959-8013-b0f3beffd4e6",          
            "140347b2-29a2-4a5f-b44f-aa293a02d9ff",
            "6b3eb082-2d0c-4daf-8edb-0b10fb3621c8",                
            "8c59732e-c1ab-4150-aff1-7e97089e6c9b",
            "5b9f594f-891e-4d57-b21d-8716e2f06c4b",
            "428af957-8413-46c4-91f5-73b5c94034bf",
            "c90fd79d-485b-4667-967a-8d0ad0c9d84b"          
        ]

        self.camera_stream_fetchers = []
        for camera_config_dict in self.camera_configs_list:
            uuid = camera_config_dict["uuid"]
            channel = camera_config_dict["channel"]
            NVR_ip = camera_config_dict["NVR_ip"]
            camera_ip_address = camera_config_dict["camera_ip_address"]
            username = camera_config_dict["username"]
            password = camera_config_dict["password"]
            stream_path = camera_config_dict["stream_path"]

            if uuid not in self.uuids_to_consider:
                continue
            
            delay_interval_between_frames = [1, 5]
            if uuid in no_delay_cemara_uuids:
                delay_interval_between_frames = [0.25,0.50]
            camera_stream_fetcher = CameraStreamFetcher(delay_interval_between_frames = delay_interval_between_frames, NVR_ip = NVR_ip, channel=channel, camera_uuid = uuid, username = username, password = password, ip_address = camera_ip_address, stream_path = stream_path, VERBOSE = self.VERBOSE)
            self.camera_stream_fetchers.append(camera_stream_fetcher)
            

    def start_watching_all_IP_cameras(self):
        for camera_stream_fetcher in self.camera_stream_fetchers:   
            if self.VERBOSE: print(f'Starting to watch {camera_stream_fetcher}') 
            camera_stream_fetcher.start_watching_IP_camera()

    def stop_watching_all_IP_cameras(self):
        for camera_stream_fetcher in self.camera_stream_fetchers:   
            if self.VERBOSE: print(f'Stopping to watch {camera_stream_fetcher}') 
            camera_stream_fetcher.stop_watching_IP_camera()

    def get_latest_frame_details(self, max_age:float = 30.0):
        latest_frames = []
        for camera_stream_fetcher in self.camera_stream_fetchers:
            if camera_stream_fetcher.get_how_old_is_the_latest_frame() > 30:
                continue
            latest_frames.append(camera_stream_fetcher.get_fetched_frame_details())

        if self.VERBOSE: print(f'\n================================================\nGot {len(latest_frames)} frames from IP cameras which are not older than {max_age} seconds.\n================================================\n')

        return latest_frames

class CameraStreamFetcher: 
    def __init__(self, camera_uuid:str = None, username:str = None, password:str = None, ip_address:str = None, stream_path:str = None, channel:str=None, NVR_ip:str=None, delay_interval_between_frames:list[float] = [0.25, 5], VERBOSE:bool=True)->None:
        self.camera_uuid = camera_uuid
        self.username = username
        self.password = password
        self.ip_address = ip_address
        self.stream_path = stream_path
        self.channel = channel #CH13 for example 
        self.NVR_ip = NVR_ip
        self.VERBOSE = VERBOSE

        self.running = False
        self.delay_between_frames = delay_interval_between_frames #delay between frames in seconds 
        self.latest_frame = None
        self.latest_frame_timestamp = 0

    def __str__ (self):
        return f'IPCameraWatcher object: {self.camera_uuid} ({self.ip_address})'
    
    def __repr__ (self):
        return f'IPCameraWatcher object: {self.camera_name} ({self.ip_address})'

    def start_watching_IP_camera(self):
        if self.VERBOSE: print(f'Starting to watch {self.ip_address} at {time.time()}')

        self.running = True
        self.thread = threading.Thread(target=self.IP_camera_frame_fetching_thread)
        self.thread.daemon = True  # Set the thread as a daemon means that it will stop when the main program stops
        self.thread.start()

    def IP_camera_frame_fetching_thread(self):
        url = f'rtsp://{self.username}:{self.password}@{self.ip_address}/{self.stream_path}'
        cap = cv2.VideoCapture(url)

        buffer_size_in_frames = 1
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size_in_frames)

        pprint.pprint(f'Watching {self.ip_address} at {time.time()}')

        last_capture_time = time.time()
        dynamic_delay = random.uniform(0, 1)
        while self.running:            
            if not cap.grab():# Use grab() to capture the frame but not decode it yet
                continue 
            
            if time.time() - last_capture_time > dynamic_delay:
                # Introduce a delay. Note that if the delay is same for all, the frames will be tried to be retrieved at the same time (almost) so it will become useless. Uniform Random delay is introduced to avoid this.
                
                min_delay = self.delay_between_frames[0]
                max_delay = self.delay_between_frames[1]
                dynamic_delay = random.uniform(min_delay, max_delay)
                last_capture_time = time.time()
                
                # Then retrieve the frame
                ret, frame = cap.retrieve()
                if ret:
                    if self.VERBOSE: print(f'Got a frame from {self.ip_address} at {time.time()}')
                    self.latest_frame = frame
                    self.latest_frame_timestamp = time.time()
                else:
                    continue

        cap.release()

    def stop_watching_IP_camera(self):
        self.running = False
        self.thread.join()

    def is_watching(self):
        return self.running

    def get_uuid(self):
        return self.camera_uuid
    
    def get_latest_frame(self):
        return self.latest_frame
    
    def get_latest_frame_timestamp(self):
        return self.latest_frame_timestamp
    
    def get_how_old_is_the_latest_frame(self) -> float:
        return time.time() - self.latest_frame_timestamp
    
    def get_fetched_frame_details(self):
        #uuid, frame, ip_address, channel, NVR_ip
        return {
            'uuid': self.camera_uuid,
            'frame': self.latest_frame,
            'ip_address': self.ip_address,
            'channel': self.channel,
            'NVR_ip': self.NVR_ip,
            'fetch_timestamp': self.latest_frame_timestamp
        }