import cv2
import time 
       
if __name__ == "__main__":
    username = input("Enter username: ")
    password = input("Enter password: ")
    ip_address = input("Enter IP address: ")

    url = f'rtsp://{username}:{password}@{ip_address}/{"profile2/media.smp"}'
    cap = cv2.VideoCapture(url)

    while True:
        
        if not cap.grab():
            print(f'Failed to connect to {ip_address}')
            continue

        ret, frame = cap.retrieve()
        if ret:
            print(f'Got a frame from {ip_address}')    
            cv2.imshow('frame', frame)           
        else:
            print(f'Failed to get a frame from {ip_address}')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break