import cv2
import matplotlib.pyplot as plt

img_array=cv2.imread('./StudyLog/OpenCV_class/OBJ9.png') # input img directory, need to change
plt.imshow(img_array)



# video capture function
video_input_path = 1 # need to change
video_output_path=2


cap = cv2.VideoCapture(video_input_path) # Video Capture is read the video per Frame
vid_writer = cv2.VideoWriter(video_output_path)

while True:
    hasFrame, img_frame = cap.read()
    if not hasFrame:
        print('nothing Frame')
        break
    