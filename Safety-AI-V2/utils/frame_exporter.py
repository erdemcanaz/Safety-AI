import cv2
import os
import uuid

image_info = input("Enter the image info (i.e CH14_forklift): ")
# Replace 'your_video_file.mp4' with the path to your video file
video_path = input("Enter the path to the video file: ")
# Specify the folder where you want to save the frames

this_script_folder = os.path.dirname(os.path.abspath(__file__))
safety_ai_folder = os.path.dirname(this_script_folder)
images_folder = os.path.join(safety_ai_folder, "images")
save_folder = images_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
frame_jump = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or can't retrieve frames.")
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + frame_jump)
    elif key == ord('e'):
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + frame_jump*15)
    elif key == ord('a'):
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - frame_jump)
    elif key == ord('q'):
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - frame_jump*15)
    elif key == ord('s'):
        # Save the current frame

        image_name = f"secret_{image_info}_{str(uuid.uuid4())}.jpg"
        frame_filename = os.path.join(save_folder, image_name)
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")
        frame_count += 1
    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
