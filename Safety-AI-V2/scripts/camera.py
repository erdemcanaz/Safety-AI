import json
import numpy as np

class Camera():
    def __init__(self, uuid:str="default"):

        with open('json_files/camera_configs.json') as f:
            camera_configs = json.load(f)["cameras"]

        self.CAMERA_CONFIGURATION = None
        for camera_config in camera_configs:
            if camera_config["uuid"] == uuid:
                self.CAMERA_CONFIGURATION = camera_config
                break
        if self.CAMERA_CONFIGURATION == None:
            raise Exception("Camera Configuration not found")

    def get_uuid(self) -> str:
        return self.CAMERA_CONFIGURATION["uuid"]
    
    def get_channel(self) -> str:
        return self.CAMERA_CONFIGURATION["channel"]
    
    def is_active(self)-> bool:
        return self.CAMERA_CONFIGURATION["is_active"]

    def get_NVR_ip(self) -> str:
        return self.CAMERA_CONFIGURATION["NVR_ip"]
    
    def get_ip(self) -> str:
        return self.CAMERA_CONFIGURATION["camera_ip_address"]
    
    def get_username(self) -> str:
        return self.CAMERA_CONFIGURATION["username"]
    
    def get_password(self) -> str:
        return self.CAMERA_CONFIGURATION["password"]
    
    def get_stream_path(self) -> str:
        return self.CAMERA_CONFIGURATION["stream_path"]
    
    def get_camera_view_angles(self)-> tuple[float, float]:
        #The camera view angles are in degrees.
        return (self.CAMERA_CONFIGURATION["camera_view_angles"]["h_angle"], self.CAMERA_CONFIGURATION["camera_view_angles"]["v_angle"])
     
    def get_camera_position_wrt_origin(self) -> np.ndarray:
        #Note that the camera position is only in 2D. The z-coordinate is assumed to be 0.        
        return np.array(matrix_T = np.array(self.CAMERA_CONFIGURATION["camera_matrices"]["T_matrix"]))
    
    def get_camera_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Matrices A and C are used to convert camera coordinates to a coordinate system identical to world frame (i.e world")
        #   in terms of rotation but located just below the the camera (i.e. z=0). The transformation is done using the following formula: 
        #   >>>> X_world" = A*X_camera + C
        #Matrix T is a vector from origin to camera in 2D.
        #   The transformation is done using the following formula: 
        #   >>>> X_world = X_world" + T

        matrix_A = np.array(self.CAMERA_CONFIGURATION["camera_matrices"]["A_matrix"])
        matrix_C = np.array(self.CAMERA_CONFIGURATION["camera_matrices"]["C_matrix"])
        matrix_T = np.array(self.CAMERA_CONFIGURATION["camera_matrices"]["T_matrix"])
        
        return (matrix_A, matrix_C, matrix_T)