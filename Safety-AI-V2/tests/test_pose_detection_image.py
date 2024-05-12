import torch
from ultralytics import YOLO
import cv2
import pprint, time

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

yolo_models = [
    'yolov8n-pose.pt',
    'yolov8s-pose.pt',
    'yolov8m-pose.pt',
    'yolov8l-pose.pt',
    'yolov8x-pose.pt',
    'yolov8x-pose-p6.pt',

]

pprint.pprint(yolo_models)

which_model_index = input("Which model do you want to use? (index):")
which_model_index = int(which_model_index)
which_model = yolo_models[which_model_index]

model = YOLO(which_model).to(device)

# Load the image
image = cv2.imread('pose_test_image.jpg')

# Predict with the model
number_of_iterations = 100
start_time = time.time()
for i in range(number_of_iterations):
    print(f"Iteration: {i}/{number_of_iterations}")
    results = model(image, show=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"FPS: {number_of_iterations/elapsed_time}")

# Close the window
cv2.destroyAllWindows()
