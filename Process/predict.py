from ultralytics import YOLO
import cv2
import sys

# Load the trained YOLO model
model = YOLO('/runs/segment/train/weights/best.pt')

# Path to the input video
video_path = 'testVideo.mp4'  

# If a video path is not provided, prompt the user to input one
if not video_path:
    video_path = input("Enter the path to the input video: ").strip()

# Run predictions on the video
results = model(video_path, save=True, stream=True)

# Define the video writer
output_path = 'outputVideo.mp4'  # Adjust path as needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
fps = 30  # Adjust based on your input video's frame rate
frame_width, frame_height = None, None

video_writer = None

for result in results:
    # `result` contains a processed frame with predictions
    frame = result.plot()  # This renders the segmentation masks on the frame

    # Initialize the video writer with the first frame's size
    if video_writer is None:
        frame_height, frame_width = frame.shape[:2]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write the frame to the output video
    video_writer.write(frame)

# Release the video writer
if video_writer is not None:
    video_writer.release()

print(f"Segmented video saved at {output_path}")
