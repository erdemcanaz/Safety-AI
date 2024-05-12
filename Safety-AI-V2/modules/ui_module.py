import cv2
import copy, random
import numpy as np
import random
import datetime 
import time 

class UIModule:
    def __init__(self):
        self.BACKGROUND_IMAGE = cv2.imread("images/src/backgrounds/ui_background_3.png")        
        self.PERSON_EMOJIS ={
            "green_forklift_hardhat": cv2.imread("images/src/icons/green_forklift_hardhat.png"),
            "green_forklift": cv2.imread("images/src/icons/green_forklift.png"),
            "green_hardhat_height": cv2.imread("images/src/icons/green_hardhat_height.png"), 
            "green_hardhat": cv2.imread("images/src/icons/green_hardhat.png"),
            "green_height": cv2.imread("images/src/icons/green_height.png"),
            "green": cv2.imread("images/src/icons/green.png"),
            "red_forklift_hardhat": cv2.imread("images/src/icons/red_forklift_hardhat.png"),
            "red_forklift": cv2.imread("images/src/icons/red_forklift.png"),
            "red_hardhat_height": cv2.imread("images/src/icons/red_hardhat_height.png"),
            "red_hardhat": cv2.imread("images/src/icons/red_hardhat.png"),
            "red_height": cv2.imread("images/src/icons/red_height.png"),
            "red": cv2.imread("images/src/icons/red.png"),
        }

        self.UUID_INFOS = {
            "94edc97e-1c91-49da-8004-f4a1b7ef1360":{
            "channel":13,
            "px_coord_wrt_origin":(175,900)
            },
            "22583b92-89c4-4b30-8bb2-a02fa7352e30":{
            "channel":14,
            "px_coord_wrt_origin":(464,880)
            },
            "b1fd15ab-fd5e-4b93-a8a7-cde4d3209ac6":{
            "channel":15,
            "px_coord_wrt_origin":(380,697)
            },
            "00d853cb-4e29-4289-8cd9-13918fefb9e7":{
            "channel":16,
            "px_coord_wrt_origin":(76,780)
            },
            "306f2a75-6240-4ae3-a930-a9b895cdcba7":{
            "channel":19,
            "px_coord_wrt_origin":(103, 434)
            },
            "47968b23-1fd5-4a96-827f-9bc1b94cfd74":{
            "channel":20,
            "px_coord_wrt_origin":(178,478)
            },
            "be06a307-c9e9-41da-a411-fa5b365e4a07":{
            "channel":21,
            "px_coord_wrt_origin":(358,478)
            },
            "94b80fe6-5fd9-481a-8cbb-00be0e1dfb1c":{
            "channel":22,
            "px_coord_wrt_origin":(478,478)
            },
            "7cabf973-f717-44a7-a261-2a3ec7cc610c":{
            "channel":23,
            "px_coord_wrt_origin":(533,427)
            },
            "92780f91-d255-41a9-acec-65af3070a7bc":{
            "channel":24,
            "px_coord_wrt_origin":(618,477)
            },           
            "d1a09fa8-80fc-4959-8013-b0f3beffd4e6":{
            "channel":25,
            "px_coord_wrt_origin":(769,477)
            },          
            "140347b2-29a2-4a5f-b44f-aa293a02d9ff":{
            "channel":26,
            "px_coord_wrt_origin":(891,477)
            },
            "6b3eb082-2d0c-4daf-8edb-0b10fb3621c8":{
            "channel":27,
            "px_coord_wrt_origin":(932,427)
            },                
            "8c59732e-c1ab-4150-aff1-7e97089e6c9b":{
            "channel":28,
            "px_coord_wrt_origin":(1013,477)
            },
            "5b9f594f-891e-4d57-b21d-8716e2f06c4b":{
            "channel":29,
            "px_coord_wrt_origin":(1087,428)
            },
            "428af957-8413-46c4-91f5-73b5c94034bf":{
            "channel":30,
            "px_coord_wrt_origin":(1168,462)
            },
            "c90fd79d-485b-4667-967a-8d0ad0c9d84b":{
            "channel":31,
            "px_coord_wrt_origin":(1258,425)
            }           
        }

        self.METER_TO_PIXEL_RATIO = 16.75 # 1 meter = 16.75 pixel
        self.ORIGIN_OFFSET_PX = (481, 905) # (x, y)
        self.UUID_IDS = {} # {camera_uuid: camera_id}

        self.update_camera_counter = 0
        self.displayed_frames = []

        self.total_person_detected = 0
        self.total_hard_hat_violation_detected = 0
        self.total_restricted_area_violation_detected = 0
        self.total_forklift_detected = 0
        self.total_person_in_restrict_area = 0
        self.total_person_in_hard_hat_area = 0

        self.camera_frame_to_show_CH14 = None
        self.camera_frame_to_show_CH15 = None

    def get_background_image(self):
        return copy.deepcopy(self.BACKGROUND_IMAGE)
    
    def get_person_emoji(self, is_violating:bool = False,  is_in_forklift:bool=False, is_wearing_hard_hat:bool = False, is_at_height:bool = False, scale_factor:float = 1):
        key_name = "red" if is_violating else "green"
        key_name = key_name + "_forklift" if is_in_forklift else key_name
        key_name = key_name + "_hardhat" if is_wearing_hard_hat else key_name
        key_name = key_name + "_height" if is_at_height else key_name

        emoji_frame = copy.deepcopy(self.PERSON_EMOJIS[key_name])      
        new_width = int(emoji_frame.shape[1] * scale_factor)
        new_height = int(emoji_frame.shape[0] * scale_factor)
        resized_emoji_frame = cv2.resize(emoji_frame, (new_width, new_height))

        return resized_emoji_frame

    def overlay_camera_14_and_15_frame(self, frame):
        new_width = 281
        new_height = 171 #161
    
        X_OFFSET = 1466
        Y_OFFSET = 175  
        
        ch14_frame_resized = None
        ch15_frame_resized = None

        if self.camera_frame_to_show_CH14 is not None:
            ch14_frame_resized = cv2.resize(self.camera_frame_to_show_CH14, (new_width, new_height))
            frame[Y_OFFSET:Y_OFFSET + new_height, X_OFFSET:X_OFFSET + new_width] = ch14_frame_resized
        if self.camera_frame_to_show_CH15 is not None:
            ch15_frame_resized = cv2.resize(self.camera_frame_to_show_CH15, (new_width, new_height))
            frame[Y_OFFSET + new_height + 10:Y_OFFSET + 2*new_height + 10, X_OFFSET:X_OFFSET + new_width] = ch15_frame_resized
          
    def overlay_time_and_shift_info(self, frame):
        # Get the current time as a timestamp
        current_timestamp = time.time()

        # Convert the timestamp to a datetime object
        month_index = time.localtime(current_timestamp).tm_mon
        day = time.localtime(current_timestamp).tm_mday
        year = time.localtime(current_timestamp).tm_year
        hour = time.localtime(current_timestamp).tm_hour
        minute = time.localtime(current_timestamp).tm_min
        second = time.localtime(current_timestamp).tm_sec

        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        formatted_string_date = f"{months[month_index - 1]} {day}, {year} - {hour}:{minute}:{second}"

        shift_index = hour // 8
        shift_infos = ["Shift I", "Shift II", "Shift III"]
        formatted_string_shift = shift_infos[shift_index]
        
        formatted_string = formatted_string_date + "                 "+ formatted_string_shift
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (1350, 600)
        fontScale = 0.65
        fontColor = (155, 7, 7)
        lineType = 2

        cv2.putText(frame, formatted_string,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType) 
               

        bar_width = 500
        bar_height = 10

        day_second = hour * 3600 + minute * 60 + second
        bar_complated_width = int((day_second % 28800) / 28800 * bar_width) #8 hour is 28800 seconds

        bar_x = 1358
        bar_y = 631
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_complated_width, bar_y + bar_height), (155, 7, 7), -1)

    def overlay_total_counts(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (500, 750)
        fontScale = 1
        fontColor = (155, 7, 7)
        lineType = 2
        cv2.putText(frame, "Total Person Count: " + str(self.total_person_detected),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType) 

        bottomLeftCornerOfText = (500, 800)
        cv2.putText(frame, "Total Hard-Hat Violation Count: " + str(self.total_hard_hat_violation_detected),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (500, 850)
        cv2.putText(frame, "Total in Hard-Hat Area Count: " + str(self.total_person_in_hard_hat_area),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (500, 900)
        cv2.putText(frame, "Total Restricted A. Violation Count: " + str(self.total_restricted_area_violation_detected),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (500, 950)
        cv2.putText(frame, "Total in Restricted Area Count: " + str(self.total_person_in_restrict_area),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (500, 1000)
        cv2.putText(frame, "Total Forklift Count: " + str(self.total_forklift_detected),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (1100, 750)
        cv2.putText(frame, "HARD-HAT VIOLATION:",
                    bottomLeftCornerOfText,
                    font,
                    1,
                    fontColor,
                    lineType)
        
        bottomLeftCornerOfText = (1100, 820)
        cv2.putText(frame, "% "+str(round(100*self.total_hard_hat_violation_detected/(self.total_person_in_hard_hat_area+1), 2)),
                    bottomLeftCornerOfText,
                    font,
                    2,
                    fontColor,
                    lineType)

    def overlay_camera_circles(self, frame, camera_uuid:str,  is_person:bool = False):
        center_x, center_y = self.UUID_INFOS[camera_uuid]["px_coord_wrt_origin"]
        cv2.circle(frame, (center_x, center_y), 20, (155, 15, 15), 4)
        if is_person:
            cv2.circle(frame, (center_x, center_y), 30, (154, 149, 15), 2)
        else:
            cv2.circle(frame, (center_x, center_y), 30, (155, 15, 15), 2)

    def overlay_person_emoji(self, frame, emoji_frame, person_x_px, person_y_px):
            emoji_height, emoji_width = emoji_frame.shape[:2]

            # Calculate the top-left corner of where the emoji will be placed
            start_x = person_x_px - (emoji_width  // 2)
            start_y = person_y_px - (emoji_height // 2)
            end_x = start_x + emoji_width
            end_y = start_y + emoji_height

            # Check for bounds and adjust if necessary
            if start_x < 0: start_x = 0
            if start_y < 0: start_y = 0
            if end_x > frame.shape[1]: end_x = frame.shape[1]
            if end_y > frame.shape[0]: end_y = frame.shape[0]

            # Overlay the emoji on the new_frame image
            frame[start_y:end_y, start_x:end_x] = emoji_frame[0:(end_y - start_y), 0:(end_x - start_x)]

    def overlay_counters(self, frame, counters:dict):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.75
        fontColor = (155, 7, 7)
        lineType = 2

        def k_formatter(num, dec_digits=1):
            num = float(num)
            if num < 1000:
                return str(int(num))
            else:
                return f"{num/1000:.{dec_digits}f}k"
        
        if "person_detection_count" in counters:          
            bottomLeftCornerOfText = (1432, 780)

            cv2.putText(frame, k_formatter(counters["person_detection_count"]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        
        if "forklift_detection_count" in counters:
            bottomLeftCornerOfText = (1756, 780)

            cv2.putText(frame, k_formatter(counters["forklift_detection_count"]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            
        if "number_of_frame_processed" in counters:
            bottomLeftCornerOfText = (1450, 941)        
            cv2.putText(frame, k_formatter(counters["number_of_frame_processed"]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


        #HEIGHT================================================================
        fontScale = 0.5
        def add_height_succes_failure_counts(y, succes_fail_counts):
            bottomLeftCornerOfText = (1149, y) #succes count 
            cv2.putText(frame, k_formatter(succes_fail_counts[0]),
                        bottomLeftCornerOfText,
                        font,   
                        fontScale,
                        fontColor,
                        lineType)
            
            bottomLeftCornerOfText = (1197, y) # fail count
            cv2.putText(frame, k_formatter(succes_fail_counts[1]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            
            bottomLeftCornerOfText = (1247, y) # violation percentage
            violation_percentage = 100*succes_fail_counts[0]/(succes_fail_counts[0]+succes_fail_counts[1]) if succes_fail_counts[0]+succes_fail_counts[1] > 0 else 0
            violation_percentage = f"{violation_percentage:.1f}"
            cv2.putText(frame, k_formatter(violation_percentage),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
                               
        if "warehouse_height_succes_fail_counts" in counters: 
            add_height_succes_failure_counts(885, counters["warehouse_height_succes_fail_counts"])
        
        if "cardboard_height_succes_fail_counts" in counters:
            add_height_succes_failure_counts(905, counters["cardboard_height_succes_fail_counts"])

        if "raw_material_height_succes_fail_counts" in counters:
            add_height_succes_failure_counts(925, counters["raw_material_height_succes_fail_counts"])

        if "allergy_height_succes_fail_counts" in counters:
            add_height_succes_failure_counts(950, counters["allergy_height_succes_fail_counts"])

        if "chiller_height_succes_fail_counts" in counters:
            add_height_succes_failure_counts(971, counters["chiller_height_succes_fail_counts"])   

        def add_height_bars(x, normalized_val):
            PEAK_BAR_HEIGHT = 115
            BAR_WIDTH = 20

            BAR_Y_OFFSET = 781
            BAR_X_LEFT = x
            BAR_X_RIGHT = x + BAR_WIDTH

            normalized_bar_height = int(normalized_val * PEAK_BAR_HEIGHT)
            cv2.rectangle(frame, (BAR_X_LEFT, BAR_Y_OFFSET), (BAR_X_RIGHT, BAR_Y_OFFSET - int(normalized_val * PEAK_BAR_HEIGHT)), (154, 15, 15), -1)
            
        total_height_counts = [0,0,0,0,0]

        if "warehouse_height_counts" in counters:
            for i in range(5):
                total_height_counts[i] += counters["warehouse_height_counts"][i]

        if "cardboard_height_counts" in counters:
            for i in range(5):
                total_height_counts[i] += counters["cardboard_height_counts"][i]

        if "raw_material_height_counts" in counters:
            for i in range(5):
                total_height_counts[i] += counters["raw_material_height_counts"][i]

        if "allergy_height_counts" in counters:
            for i in range(5):
                total_height_counts[i] += counters["allergy_height_counts"][i]

        if "chiller_height_counts" in counters:
            for i in range(5):
                total_height_counts[i] += counters["chiller_height_counts"][i]

        max_height_count = 1
        for i in range(5):
            max_height_count = max(total_height_counts[i], max_height_count)

        add_height_bars(1056, total_height_counts[0]/max_height_count)
        add_height_bars(1106, total_height_counts[1]/max_height_count)
        add_height_bars(1146, total_height_counts[2]/max_height_count)
        add_height_bars(1187, total_height_counts[3]/max_height_count)
        add_height_bars(1227, total_height_counts[4]/max_height_count)       
          
        #HARD-HAT================================================================
        def add_hard_hat_succes_failure_counts(y, hard_hat_counts):
            bottomLeftCornerOfText = (635, y) #succes count 
            cv2.putText(frame, k_formatter(hard_hat_counts[0]),
                        bottomLeftCornerOfText,
                        font,   
                        fontScale,
                        fontColor,
                        lineType)
            
            bottomLeftCornerOfText = (685, y) # fail count
            cv2.putText(frame, k_formatter(hard_hat_counts[1]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            
            bottomLeftCornerOfText = (734, y) # violation percentage
            violation_percentage = 100*hard_hat_counts[0]/(hard_hat_counts[0]+hard_hat_counts[1]) if hard_hat_counts[0]+hard_hat_counts[1] > 0 else 0
            violation_percentage = f"{violation_percentage:.1f}"
            cv2.putText(frame, k_formatter(violation_percentage),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


        total_hard_hat_counts = [0,0]
        if "warehouse_hard_hat_succes_fail_counts" in counters:
            add_hard_hat_succes_failure_counts(887, counters["warehouse_hard_hat_succes_fail_counts"])
            total_hard_hat_counts[0] += counters["warehouse_hard_hat_succes_fail_counts"][0]
            total_hard_hat_counts[1] += counters["warehouse_hard_hat_succes_fail_counts"][1]

        if "cardboard_hard_hat_succes_fail_counts" in counters:
            add_hard_hat_succes_failure_counts(907, counters["cardboard_hard_hat_succes_fail_counts"])
            total_hard_hat_counts[0] += counters["cardboard_hard_hat_succes_fail_counts"][0]
            total_hard_hat_counts[1] += counters["cardboard_hard_hat_succes_fail_counts"][1]

        if "raw_material_hard_hat_succes_fail_counts" in counters:
            add_hard_hat_succes_failure_counts(927, counters["raw_material_hard_hat_succes_fail_counts"])
            total_hard_hat_counts[0] += counters["raw_material_hard_hat_succes_fail_counts"][0]
            total_hard_hat_counts[1] += counters["raw_material_hard_hat_succes_fail_counts"][1]

        if "allergy_hard_hat_succes_fail_counts" in counters:
            add_hard_hat_succes_failure_counts(952, counters["allergy_hard_hat_succes_fail_counts"])
            total_hard_hat_counts[0] += counters["allergy_hard_hat_succes_fail_counts"][0]
            total_hard_hat_counts[1] += counters["allergy_hard_hat_succes_fail_counts"][1]

        if "chiller_hard_hat_succes_fail_counts" in counters:
            add_hard_hat_succes_failure_counts(973, counters["chiller_hard_hat_succes_fail_counts"])
            total_hard_hat_counts[0] += counters["chiller_hard_hat_succes_fail_counts"][0]
            total_hard_hat_counts[1] += counters["chiller_hard_hat_succes_fail_counts"][1]

        #create pie chart for hard hat violation
        val_1 = total_hard_hat_counts[0]
        val_2 = total_hard_hat_counts[1]

        center = (645, 725)
        radius = 55

        cv2.circle(frame, center, radius, (155, 7, 7), -1)        
        cv2.ellipse(frame, center, (radius, radius), 0, 0, int(360*val_1/(max(val_1+val_2,1))), (154, 149, 15), -1)
        cv2.circle(frame, center, radius-10, (247, 240, 228), -1)

        succes_percentage = 100*val_1/(val_1+val_2) if val_1+val_2 > 0 else 0
        cv2.putText(frame, f"{succes_percentage:.1f}%", (615, 732), font, 1, (155, 7, 7), 2)

        #ENERGY CONSUMPTION =============================================
        if "kWh_energy_per_million_frames" in counters:
            bottomLeftCornerOfText = (1737, 911)
            fontColor = (82, 190, 0)

            kWh_formatted = f"{counters['kWh_energy_per_million_frames']:.2f}"    
            if counters['kWh_energy_per_million_frames'] > 100:
                kWh_formatted = f">100"

            cv2.putText(frame,kWh_formatted,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
         
    def update_ui_frame(self, window_name:str= "default-window", counters = {}, multiple_camera_evaluation_results: list[dict] =None, wait_time_ms:int = 0, window_scale_factor:float = 1, emoji_scale_factor = 1.5, timestamp_str = None)-> bool:
        # Draw the UI frame
        new_frame = self.get_background_image()

        self.overlay_counters(new_frame, counters)

        for evaluation_results in multiple_camera_evaluation_results:
            camera_uuid= evaluation_results["camera_uuid"]

            number_of_forklifts = evaluation_results["number_of_forklifts"]            
            number_of_persons = evaluation_results["number_of_persons"]
            person_evaluations = evaluation_results["person_evaluations"]              
       
            self.total_person_detected += number_of_persons
            self.total_forklift_detected += number_of_forklifts

            if number_of_forklifts > 0:
                is_person = True if number_of_persons > 0 else False
                self.overlay_camera_circles(new_frame, camera_uuid, is_person = is_person )

            for person_evaluation in person_evaluations:
                is_in_forklift = person_evaluation["is_in_forklift"]
                is_wearing_hard_hat = person_evaluation["is_wearing_hard_hat"]
                is_at_height = person_evaluation["is_at_height"]

                is_violating_restricted_area = person_evaluation["is_violating_restricted_area_rule"]
                is_violating_hard_hat_area = person_evaluation["is_violating_hard_hat_rule"]
                is_violating_height_rule = person_evaluation["is_violating_height_rule"]                
                is_violating = is_violating_restricted_area or is_violating_hard_hat_area or is_violating_height_rule
                

                if is_violating_hard_hat_area:
                    self.total_hard_hat_violation_detected += 1

                if is_violating_restricted_area:
                    self.total_restricted_area_violation_detected += 1

                is_in_hard_hat_area = person_evaluation["is_in_hard_hat_rule_area"]
                if is_in_hard_hat_area:
                    self.total_person_in_hard_hat_area += 1
                if is_violating_restricted_area:
                    self.total_person_in_restrict_area += 1


                person_x = person_evaluation["world_coordinate"][0][0]
                person_y = person_evaluation["world_coordinate"][1][0]

                person_x_px = int(person_x * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[0]
                person_y_px = int(person_y * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[1]       

                emoji_scaler = 1.25*emoji_scale_factor if is_violating else emoji_scale_factor       
                emoji_frame = self.get_person_emoji(scale_factor= emoji_scaler, is_violating=is_violating, is_in_forklift=is_in_forklift, is_wearing_hard_hat=is_wearing_hard_hat, is_at_height=is_at_height)
                self.overlay_person_emoji(new_frame, emoji_frame, person_x_px, person_y_px)

            if evaluation_results['camera_uuid'] == "22583b92-89c4-4b30-8bb2-a02fa7352e30": #CH14
                self.camera_frame_to_show_CH14 = evaluation_results["frame_externally_added_key"]
            if evaluation_results["camera_uuid"] == "b1fd15ab-fd5e-4b93-a8a7-cde4d3209ac6": #CH15
                self.camera_frame_to_show_CH15 = evaluation_results["frame_externally_added_key"]
         
        self.overlay_camera_14_and_15_frame(new_frame)

        #self.overlay_total_counts(new_frame)
        if timestamp_str != None:
            self.overlay_time_and_shift_info(new_frame)

        new_width = int(new_frame.shape[1] * window_scale_factor)
        new_height = int(new_frame.shape[0] * window_scale_factor)
        resized_new_frame = cv2.resize(new_frame, (new_width, new_height))
        cv2.imshow(window_name, resized_new_frame)

        # Wait for a key press to close the window
        key = cv2.waitKey(wait_time_ms)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return True, resized_new_frame
        return False, resized_new_frame
     
