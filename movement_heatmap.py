import cv2
import numpy as np

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error accessing webcam"

# Get properties from the video capture
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Might not be accurate for live feeds and can be set manually

# Create Background Subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Initialize an empty image for accumulating the heat map
heatmap = np.zeros((h, w), dtype=np.float32)

# Define the decay factor (try different values to see what works best for your case)
decay_factor = 0.99  # This value should be close to but less than 1

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the frame is not successfully captured

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Use the foreground mask to update the heatmap
    heatmap = cv2.add(heatmap, fgmask.astype(np.float32))

    # Apply the decay to the heatmap
    heatmap *= decay_factor  # Reducing the intensity of the heatmap gradually

    # Normalize the heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert heatmap to BGR to display
    heatmap_bgr = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Display the original frame and the heatmap
    cv2.imshow('Frame', frame)
    cv2.imshow('Heatmap', heatmap_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy all windows
cap.release()
cv2.destroyAllWindows()
