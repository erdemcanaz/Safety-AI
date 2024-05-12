import time
import pprint

class CounterModule():

    def __init__(self):

        self.PROGRAM_START_TIME = time.time()
        self.shift_no = 0 # 0 for 00:00-08:00, 1 for 08:00-16:00, 2 for 16:00-00:00
        self.reset_counters()
        
    def reset_counters(self):

        self.person_detection_count = 0
        self.forklift_detection_count = 0
        self.number_of_frame_processed = 0


        #hard hat violation counters
        self.warehouse_hard_hat_succes_fail_counts = [0,0]
        self.cardboard_hard_hat_succes_fail_counts = [0,0]
        self.raw_material_hard_hat_succes_fail_counts = [0,0]
        self.allergy_hard_hat_succes_fail_counts = [0,0]
        self.chiller_hard_hat_succes_fail_counts = [0,0]

        #restricted area violation counters
        self.restricted_area_violation_count = 0

        #height violation counters    
        self.warehouse_height_counts = [0,0,0,0,0]
        self.cardboard_height_counts = [0,0,0,0,0]
        self.raw_material_height_counts = [0,0,0,0,0]
        self.allergy_height_counts = [0,0,0,0,0]
        self.chiller_height_counts = [0,0,0,0,0]

        self.warehouse_height_succes_fail_counts = [0,0]
        self.cardboard_height_succes_fail_counts = [0,0]
        self.raw_material_height_succes_fail_counts = [0,0]
        self.allergy_height_succes_fail_counts = [0,0]
        self.chiller_height_succes_fail_counts = [0,0]

        #energy consumption
        self.kWh_energy_per_million_frames = 0
        
    def update_counters(self, evaluation_results:dict)->None:

        # Get the current time as a timestamp
        current_timestamp = time.time()

        # Convert the timestamp to a datetime object
        hour = time.localtime(current_timestamp).tm_hour
        minute = time.localtime(current_timestamp).tm_min
        second = time.localtime(current_timestamp).tm_sec

        shift = hour//8
        if shift != self.shift_no:
            self.reset_counters()
            self.shift_no = shift

        #generic counters
        self.person_detection_count += evaluation_results["number_of_persons"]
        self.forklift_detection_count += evaluation_results["number_of_forklifts"]
        self.number_of_frame_processed += 1
      
        warehouse_camera_uuids = [ #CH 13, CH 14, CH 15, CH 16 
            "94edc97e-1c91-49da-8004-f4a1b7ef1360",
            "22583b92-89c4-4b30-8bb2-a02fa7352e30",
            "b1fd15ab-fd5e-4b93-a8a7-cde4d3209ac6",
             "00d853cb-4e29-4289-8cd9-13918fefb9e7"
        ]

        cardboard_camera_uuids = [ #CH 19, CH20, CH 21, CH 22, CH 23
            "306f2a75-6240-4ae3-a930-a9b895cdcba7",
            "47968b23-1fd5-4a96-827f-9bc1b94cfd74",
            "be06a307-c9e9-41da-a411-fa5b365e4a07",
            "94b80fe6-5fd9-481a-8cbb-00be0e1dfb1c",
            "7cabf973-f717-44a7-a261-2a3ec7cc610c"
        ]

        raw_material_camera_uuids = [ #CH 24, CH 25, CH 26, CH 27
            "92780f91-d255-41a9-acec-65af3070a7bc",
            "d1a09fa8-80fc-4959-8013-b0f3beffd4e6",
            "140347b2-29a2-4a5f-b44f-aa293a02d9ff",
            "6b3eb082-2d0c-4daf-8edb-0b10fb3621c8"
        ]

        allergy_camera_uuids = [ #CH 28, CH 29
            "8c59732e-c1ab-4150-aff1-7e97089e6c9b",
            "5b9f594f-891e-4d57-b21d-8716e2f06c4b"
        ]            

        person_evaluation_dicts = evaluation_results["person_evaluations"]


        #HEIGHT ========================================================================================================
        #height violation counters
        # [0] -> up to 1.75m
        # [1] -> 1.75m to 2.25m
        # [2] -> 2.25m to 2.75m
        # [3] -> 2.75m to 3.25
        # [4] -> 3.25m to infinity
        for person_evaluation_dict in person_evaluation_dicts:
            person_z = person_evaluation_dict["world_coordinate"][2]
            
            is_at_height = person_evaluation_dict["is_at_height"]  
            is_in_forklift = person_evaluation_dict["is_in_forklift"]

            index_to_increment = None
        
            if person_z <= 1.75:
                index_to_increment = 0
            elif person_z <= 2.25:
                index_to_increment = 1
            elif person_z <= 2.75:
                index_to_increment = 2
            elif person_z <= 3.25:
                index_to_increment = 3
            else:
                index_to_increment = 4


            if evaluation_results["camera_uuid"] in warehouse_camera_uuids:    
                if not is_in_forklift:          
                    self.warehouse_height_counts[index_to_increment] += 1
                if not is_at_height:
                    self.warehouse_height_succes_fail_counts[0] += 1
                else:
                    self.warehouse_height_succes_fail_counts[1] += 1

            elif evaluation_results["camera_uuid"] in cardboard_camera_uuids:
                self.cardboard_height_counts[index_to_increment] += 1
                if not is_at_height:
                    self.cardboard_height_succes_fail_counts[0] += 1
                else:
                    self.cardboard_height_succes_fail_counts[1] += 1
            
            elif evaluation_results["camera_uuid"] in raw_material_camera_uuids:
                self.raw_material_height_counts[index_to_increment] += 1
                if not is_at_height:
                    self.raw_material_height_succes_fail_counts[0] += 1
                else:
                    self.raw_material_height_succes_fail_counts[1] += 1
            
            elif evaluation_results["camera_uuid"] in allergy_camera_uuids:
                self.allergy_height_counts[index_to_increment] += 1
                if not is_at_height:
                    self.allergy_height_succes_fail_counts[0] += 1
                else:
                    self.allergy_height_succes_fail_counts[1] += 1

            else:
                self.chiller_height_counts[index_to_increment] += 1
                if not is_at_height:
                    self.chiller_height_succes_fail_counts[0]+=1
                else:
                    self.chiller_height_succes_fail_counts[1]+=1
        
        #HARD HAT ========================================================================================================
        for person_evaluation_dict in person_evaluation_dicts:        
            is_in_hard_hat_rule_area = person_evaluation_dict["is_in_hard_hat_rule_area"]
            if not is_in_hard_hat_rule_area:
                continue

            is_violating_hard_hat = person_evaluation_dict["is_violating_hard_hat_rule"] 

            if evaluation_results["camera_uuid"] in warehouse_camera_uuids:
                if not is_violating_hard_hat:
                    self.warehouse_hard_hat_succes_fail_counts[0] += 1
                else:
                    self.warehouse_hard_hat_succes_fail_counts[1] += 1

            elif evaluation_results["camera_uuid"] in cardboard_camera_uuids:
                if not is_violating_hard_hat:
                    self.cardboard_hard_hat_succes_fail_counts[0] += 1
                else:
                    self.cardboard_hard_hat_succes_fail_counts[1] += 1

            elif evaluation_results["camera_uuid"] in raw_material_camera_uuids:
                if not is_violating_hard_hat:
                    self.raw_material_hard_hat_succes_fail_counts[0] += 1
                else:
                    self.raw_material_hard_hat_succes_fail_counts[1] += 1

            elif evaluation_results["camera_uuid"] in allergy_camera_uuids:
                if not is_violating_hard_hat:
                    self.allergy_hard_hat_succes_fail_counts[0] += 1
                else:
                    self.allergy_hard_hat_succes_fail_counts[1] += 1

            else:
                if not is_violating_hard_hat:
                    self.chiller_hard_hat_succes_fail_counts[0] += 1
                else:
                    self.chiller_hard_hat_succes_fail_counts[1] += 1
                        
        #ESTIMATED POWER CONSUMPTION PER ONE MILLION FRAME
        TV_POWER_W = 100
        AGX_ORIN_POWER_W = 50

        total_power = TV_POWER_W + AGX_ORIN_POWER_W
        time_passed_since_start = current_timestamp - self.PROGRAM_START_TIME
        total_energy_kWh = (time_passed_since_start*total_power) / 3600000

        self.kWh_energy_per_million_frames = (total_energy_kWh / min(self.number_of_frame_processed,1))*1000000

    def get_counters(self):
        return {
            "person_detection_count": self.person_detection_count,
            "forklift_detection_count": self.forklift_detection_count,
            "number_of_frame_processed": self.number_of_frame_processed,

            "warehouse_height_counts": self.warehouse_height_counts,
            "cardboard_height_counts": self.cardboard_height_counts,
            "raw_material_height_counts": self.raw_material_height_counts,
            "allergy_height_counts": self.allergy_height_counts,
            "chiller_height_counts": self.chiller_height_counts,

            "warehouse_height_succes_fail_counts": self.warehouse_height_succes_fail_counts,
            "cardboard_height_succes_fail_counts": self.cardboard_height_succes_fail_counts,
            "raw_material_height_succes_fail_counts": self.raw_material_height_succes_fail_counts,
            "allergy_height_succes_fail_counts": self.allergy_height_succes_fail_counts,
            "chiller_height_succes_fail_counts": self.chiller_height_succes_fail_counts,

            "warehouse_hard_hat_succes_fail_counts": self.warehouse_hard_hat_succes_fail_counts,
            "cardboard_hard_hat_succes_fail_counts": self.cardboard_hard_hat_succes_fail_counts,
            "raw_material_hard_hat_succes_fail_counts": self.raw_material_hard_hat_succes_fail_counts,
            "allergy_hard_hat_succes_fail_counts": self.allergy_hard_hat_succes_fail_counts,
            "chiller_hard_hat_succes_fail_counts": self.chiller_hard_hat_succes_fail_counts,

            "kWh_energy_per_million_frames": self.kWh_energy_per_million_frames
            
        }
            

