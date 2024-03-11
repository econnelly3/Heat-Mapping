from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Initialize the video capture from the webcam
# Note: You might need to change the index in VideoCapture(0) if you have multiple video inputs
cap = cv2.VideoCapture(0) 
assert cap.isOpened(), "Error accessing webcam"

# Get properties from the video capture
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # This might not be accurate for live feeds and can be set manually

# Video writer setup (optional, if you want to save the output)
video_writer = cv2.VideoWriter("live_heatmap_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Initialize heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,  # Set to False if you don't want to display the output in a window
                     shape="circle")

while True:
    success, im0 = cap.read()
    if not success:
        break  # Exit the loop if the frame is not successfully captured

    # Process the frame with YOLO model for object tracking
    tracks = model.track(im0, persist=True, show=False)

    # Generate heatmap based on the tracked objects
    im0 = heatmap_obj.generate_heatmap(im0, tracks)

    # Write the processed frame to the output video file (optional)
    video_writer.write(im0)

    # Display the frame with heatmap
    cv2.imshow("Live Heatmap", im0)
    if cv2.waitKey(1) == ord('q'):  # Break the loop when 'q' is pressed
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
