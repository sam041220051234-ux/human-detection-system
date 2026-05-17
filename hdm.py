from ultralytics import YOLO
import cv2
import os

# Load a model
model = YOLO("hmd.pt")  # pretrained YOLO model

# Process an image
results = model(["image.jpg"])  # inference on image
for result in results:
    boxes = result.boxes
    result.show()
    result.save(filename="result.jpg")

# Process video files
video_files = ["video1.mp4", "video2.mp4"]

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get clean name for output
    name_only = os.path.splitext(os.path.basename(video_file))[0]
    out = cv2.VideoWriter(f"result_{name_only}.mp4", fourcc, fps, (width, height))

    # Inference stream
    results = model(video_file, stream=True)

    for result in results:
        frame = result.plot()
        out.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()