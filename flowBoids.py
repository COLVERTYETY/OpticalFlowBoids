import numpy as np
import cv2
from simple_boids import Boids
import colorsys
import cv2 as cv
import numpy as np

def generate_evenly_spaced_colors(n):
    """
    Generate a list of n evenly spaced RGB colors.
    """
    colors = []
    for i in range(n):
        hue = i / float(n)
        saturation = 1.0  # Set saturation to 1 for full color
        value = 1.0       # Set value to 1 for full brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert RGB values from 0-1 range to 0-255
        rgb_255 = tuple(int(val * 255) for val in rgb)
        colors.append(rgb_255)
    return colors
  
# The video feed is read in as
# a VideoCapture object
# cap = cv.VideoCapture("videoplayback.mp4")
cap = cv.VideoCapture(0)
  
# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()
  
# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally 
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
  
# Creates an image filled with zero
# intensities with the same dimensions 
# as the frame
mask = np.zeros_like(first_frame)
  
# Sets image saturation to maximum
mask[..., 1] = 255

max_movement = 100000
max_speed = 7
min_speed = 3
alpha = 0.1
b_width, b_height = 800, 600
b_screen = np.zeros((b_height, b_width, 3), dtype=np.uint8)
boids = Boids(100, b_width, b_height)
# colors is a dict that maps cluster labels to colors
# colors = {i: np.random.randint(0, 255, 3) for i in range(100)}
# use HSV color space to generate linearly spaced colors
colors = generate_evenly_spaced_colors(10)
# shuffle the colors
np.random.shuffle(colors)


while(cap.isOpened()):
      
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()
    if ret == False:
        continue
    # Opens a new window and displays the input
    # frame
    # cv.imshow("input", frame)
      
    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
      
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
      
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
      
    # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)

    # print(magnitude.shape, np.min(magnitude), np.max(magnitude))
    binarized = np.log(magnitude+1) 
    cv.imshow("binarized", binarized.astype(np.uint8) * 255)
    movement = np.sum(binarized) / max_movement
    speed = min_speed + movement * (max_speed - min_speed)
    speed = alpha * speed + (1 - alpha) * boids.speed
    boids.speed = speed
    # print("movement: ", movement, "speed: ", speed)
    boids.update(b_width, b_height)
    b_screen.fill(0)
    labels = boids.cluster()
    non_nan_positions = boids.position[~np.isnan(boids.position).any(axis=1)]
    non_nan_labels = labels[~np.isnan(boids.position).any(axis=1)]
    for position, label in zip(non_nan_positions, labels):
        label = label % len(colors)
        cv2.circle(b_screen, (int(position[0]), int(position[1])), (int)(1+movement), colors[label], -1)
    cv2.imshow("Boids Murmuration Simulation", b_screen)
    # Updates previous frame
    prev_gray = gray
      
    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()