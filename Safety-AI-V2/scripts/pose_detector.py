from ultralytics import YOLO
import cv2,math,time,os
import time

from scripts.camera import Camera

from scipy.optimize import minimize
import numpy as np

class PoseDetector(): 
    #keypoints detected by the model in the detection order
    KEYPOINT_NAMES = ["left_eye", "rigt_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    #approximate distances between the keypoints of a person in meters (1.75m)
    SHOULDER_TO_SHOULDER = 0.36 
    SHOULDER_TO_HIP = 0.48 
    SHOULDER_TO_COUNTER_HIP = 0.53  
    SHOULDER_TO_ELBOW = 0.26

    def __init__(self, model_path : str ) -> None:      
        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH, verbose= False)        
        self.recent_prediction_results = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def get_empty_prediction_dict_template(self) -> dict:
        empty_prediction_dict = {   
                    "DETECTOR_TYPE":"PoseDetector",                             # which detector made this prediction
                    "frame_shape": [0,0],                                       # [0,0], [height , width] in pixels
                    "class_name":"",                                            # hard_hat, no_hard_hat
                    "bbox_confidence":0,                                        # 0.0 to 1.0
                    "bbox_xyxy_px":[0,0,0,0],                                   # [x1,y1,x2,y2] in pixels
                    "bbox_center_px": [0,0],                                    # [x,y] in pixels

                    #------------------pose specific fields------------------
                    "is_coordinated_wrt_camera": False,                         # True if the coordinates are wrt the camera, False if they are wrt the frame
                    "belly_coordinate_wrt_camera": np.array([[0],[0],[0]]),     # [x,y,z] coordinates of the object wrt the camera
                    "is_coordinated_wrt_world_frame": False,
                    "belly_coordinate_wrt_world_frame":np.array([[0],[0],[0]]),
                    "keypoints": {                                              # Keypoints are in the format [x,y,confidence,x_angle, y_angle]
                        "left_eye": [0,0,0,0,0],
                        "right_eye": [0,0,0,0,0],
                        "nose": [0,0,0,0,0],
                        "left_ear": [0,0,0,0,0],
                        "right_ear": [0,0,0,0,0],
                        "left_shoulder": [0,0,0,0,0],
                        "right_shoulder": [0,0,0,0,0],
                        "left_elbow": [0,0,0,0,0],
                        "right_elbow": [0,0,0,0,0],
                        "left_wrist": [0,0,0,0,0],
                        "right_wrist": [0,0,0,0,0],
                        "left_hip": [0,0,0,0,0],
                        "right_hip": [0,0,0,0,0],
                        "left_knee": [0,0,0,0,0],
                        "right_knee": [0,0,0,0,0],
                        "left_ankle": [0,0,0,0,0],
                        "right_ankle": [0,0,0,0,0],
                    }
        }
        return empty_prediction_dict
    
    def predict_frame_and_return_detections(self, frame, camera_object:Camera = None) -> list[dict]:
        self.recent_prediction_results = []
        
        results = self.yolo_object(frame, task = "pose", verbose= False)[0]
        for i, result in enumerate(results):
            boxes = result.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            if box_cls_name not in ["person"]:
                continue
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]

            prediction_dict_template = self.get_empty_prediction_dict_template()
            prediction_dict_template["frame_shape"] = list(results.orig_shape)
            prediction_dict_template["class_name"] = box_cls_name
            prediction_dict_template["bbox_confidence"] = box_conf
            prediction_dict_template["bbox_xyxy_px"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            prediction_dict_template["bbox_center_px"] = [ (box_xyxy[0]+box_xyxy[2])/2, (box_xyxy[1]+box_xyxy[3])/2]
            
            key_points = result.keypoints  # Keypoints object for pose outputs
            keypoint_confs = key_points.conf.cpu().numpy()[0]
            keypoints_xy = key_points.xy.cpu().numpy()[0]
                       
            frame_height = prediction_dict_template['frame_shape'][0]
            frame_width = prediction_dict_template['frame_shape'][1]
            h_angle, v_angle = camera_object.get_camera_view_angles()
            for keypoint_index, keypoint_name in enumerate(PoseDetector.KEYPOINT_NAMES):
                keypoint_conf = keypoint_confs[keypoint_index] 
                keypoint_x = keypoints_xy[keypoint_index][0]
                keypoint_y = keypoints_xy[keypoint_index][1]
                if keypoint_x == 0 and keypoint_y == 0: #if the keypoint is not detected
                    #But this is also a prediction. Thus the confidence should not be set to zero. negative values are used to indicate that the keypoint is not detected
                    keypoint_conf = -keypoint_conf

                x_angle = ((keypoint_x/frame_width)-0.5)*h_angle
                y_angle = (0.5-(keypoint_y/frame_height))*v_angle

                prediction_dict_template["keypoints"][keypoint_name] = [keypoint_x, keypoint_y , keypoint_conf, x_angle, y_angle]

            self.approximate_prediction_distance(prediction_dict=prediction_dict_template, camera_object=camera_object)
                
            self.recent_prediction_results.append(prediction_dict_template)
        
        return self.recent_prediction_results
    
    def approximate_prediction_distance(self, prediction_dict:dict= None, camera_object:Camera = None):
        """
        Calculates the distances between the camera and each detected person. if shoulders and hips are detected

        box_condifence_threshold: minimum confidence of the bounding box to be considered while calculating distance
        distance_threshold: minimum distance that the belly of the person should be away from the camera to be considered while calculating distance in meters
        """
        DISTANCE_THRESHOLD = 1 # Min. distance in meters from camera to belly in meters
        SHOULDERS_CONFIDENCE_THRESHOLD = 0.75 # Min. confidence of the shoulders to be considered while calculating distance
        f_get_unit_vector = lambda angle_x, angle_y: [math.cos(math.radians(angle_y))*math.sin(math.radians(angle_x)), math.sin(math.radians(angle_y)), math.cos(math.radians(angle_y))* math.cos(math.radians(angle_x))]            
        
        rs_data = prediction_dict["keypoints"]["right_shoulder"]
        ls_data = prediction_dict["keypoints"]["left_shoulder"]
        rh_data = prediction_dict["keypoints"]["right_hip"]
        lh_data = prediction_dict["keypoints"]["left_hip"]

        if rs_data[2] < SHOULDERS_CONFIDENCE_THRESHOLD or ls_data[2] < SHOULDERS_CONFIDENCE_THRESHOLD:
            #to calculate distance, it is necessary to have the two shoulder keypoints by this algorithm
            return
        
        rs_uv = f_get_unit_vector(rs_data[3], rs_data[4])
        ls_uv = f_get_unit_vector(ls_data[3], ls_data[4])
        rh_uv = f_get_unit_vector(rh_data[3], rh_data[4])
        lh_uv = f_get_unit_vector(lh_data[3], lh_data[4])

        def error_function(unknowns, unit_vectors)-> float:
            k_rs,k_ls,k_rh, k_lh = unknowns
            u_rs, u_ls, u_rh, u_lh = unit_vectors

            f_scale_vector = lambda vector, scale: [vector[0]*scale, vector[1]*scale, vector[2]*scale]
            v_rs = f_scale_vector(u_rs, k_rs)
            v_ls = f_scale_vector(u_ls, k_ls)
            v_rh = f_scale_vector(u_rh, k_rh)
            v_lh = f_scale_vector(u_lh, k_lh)


            f_distance_between_vectors = lambda v1, v2: math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
            d_rs_ls = f_distance_between_vectors(v_rs, v_ls)
            d_rs_lh = f_distance_between_vectors(v_rs, v_lh)
            d_rs_rh = f_distance_between_vectors(v_rs, v_rh)
            d_ls_lh = f_distance_between_vectors(v_ls, v_lh)
            d_ls_rh = f_distance_between_vectors(v_ls, v_rh)
            
            #rs,ls,rh triangle error
            error_1 =  (d_rs_ls - PoseDetector.SHOULDER_TO_SHOULDER)**2 + (d_rs_rh - PoseDetector.SHOULDER_TO_HIP)**2 + (d_ls_rh - PoseDetector.SHOULDER_TO_COUNTER_HIP)**2
            #rs,ls,lh triangle error
            error_2 = (d_rs_ls - PoseDetector.SHOULDER_TO_SHOULDER)**2 + (d_ls_lh - PoseDetector.SHOULDER_TO_HIP)**2 + (d_rs_lh - PoseDetector.SHOULDER_TO_COUNTER_HIP)**2
            error = error_1 + error_2

            return error
        
        #optimize the triangle
        tolerance = 1e-6
        initial_guess = [5,5,5,5] #each unit vector is multiplied by a scalar
        unit_vectors = [rs_uv, ls_uv, rh_uv, lh_uv]
        #NOTE: never remove comma after unit_vectors. Othewise it will be interpreted as a tuple
        minimizer_result = minimize(error_function, initial_guess, args=( unit_vectors, ), method='L-BFGS-B', tol=tolerance)

        if minimizer_result.success == True:
            # k_rs, k_ls, k_rh, k_lh = unknowns
            scalars = minimizer_result.x
            v_rs = [rs_uv[0]*scalars[0], rs_uv[1]*scalars[0], rs_uv[2]*scalars[0]]
            v_ls = [ls_uv[0]*scalars[1], ls_uv[1]*scalars[1], ls_uv[2]*scalars[1]]
            v_rh = [rh_uv[0]*scalars[2], rh_uv[1]*scalars[2], rh_uv[2]*scalars[2]]
            v_lh = [lh_uv[0]*scalars[3], lh_uv[1]*scalars[3], lh_uv[2]*scalars[3]]

            v_camera_to_belly = None
            d_camera_to_belly = None
            if rh_data[2]>lh_data[2]:
                #
                v_camera_to_belly = [ (v_ls[0]+v_rh[0])/2, (v_ls[1]+v_rh[1])/2, (v_ls[2]+v_rh[2])/2 ]
                d_camera_to_belly = math.sqrt(v_camera_to_belly[0]**2 + v_camera_to_belly[1]**2 + v_camera_to_belly[2]**2)
            else:
                v_camera_to_belly = [ (v_rs[0]+v_lh[0])/2, (v_rs[1]+v_lh[1])/2, (v_rs[2]+v_lh[2])/2 ]
                d_camera_to_belly = math.sqrt(v_camera_to_belly[0]**2 + v_camera_to_belly[1]**2 + v_camera_to_belly[2]**2)

            #v_belly = A(world_coordinate) + C
            A, C, T = camera_object.get_camera_matrices()         

            v_camera_to_belly = np.array(v_camera_to_belly)
            v_camera_to_belly = np.reshape(v_camera_to_belly, (3,1))

            if d_camera_to_belly > DISTANCE_THRESHOLD:
                prediction_dict["is_coordinated_wrt_camera"] = True       
                prediction_dict["belly_coordinate_wrt_camera"] = v_camera_to_belly
                prediction_dict["belly_distance_wrt_camera"] = d_camera_to_belly

                world_coordinate = np.linalg.pinv(A) @ (v_camera_to_belly - C)
                world_coordinate[0][0]= world_coordinate[0][0] + T[0]
                world_coordinate[1][0]= world_coordinate[1][0] + T[1]
                world_coordinate[2][0]= world_coordinate[2][0] + T[2]
                prediction_dict["belly_coordinate_wrt_world_frame"] = world_coordinate
                prediction_dict["belly_distance_wrt_world_frame"] = math.sqrt(world_coordinate[0][0]**2 + world_coordinate[1][0]**2 + world_coordinate[2][0]**2)
                prediction_dict["is_coordinated_wrt_world_frame"] = True

   

   



