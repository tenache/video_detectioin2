from main import detect_frames
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from time import sleep
import pickle as pkl
import seaborn as sns
from numpy.ma import masked_array



TRANSPARENCIA = 0.75

video_name = "camera_02-URQUIZA_main_20230703193142.dav"
video_path = os.path.join(".","sample_data",video_name)
video_out_path = os.path.join(".","out_data", video_name + "_out.mp4")

all_detections, height, width = detect_frames(video_name, video_path, video_out_path, show_me=False, max_frames=None)

heatmap = np.zeros((height, width))

cap = cv2.VideoCapture(video_path)
ret,frame = cap.read(0)

for detection in all_detections:
    print(detection)
    x1,y1, x2, y2 = detection
    
    heatmap[y1:y2,x1:x2] += 1
    
# Normalizamos

print(f"np.max(heatmap is {np.max(heatmap)}")
print(f"heatmap {heatmap}")

heatmap = heatmap / np.max(heatmap)

heatmap_masked = masked_array(heatmap, heatmap==0)

# Desplegamos el heatmap
print(f"heatmap is {heatmap}")
with open(os.path.splitext(video_name)[0] + "pickle",'wb') as handle:
    pkl.dump(heatmap_masked, handle)


plt.imshow(frame)
plt.imshow(heatmap_masked, cmap = 'bwr', interpolation = 'nearest', alpha=TRANSPARENCIA)
plt.savefig(os.path.join("heatmaps",f"{os.path.splitext(video_name)[0]}_bwr.png"))
plt.imshow(heatmap_masked, cmap = 'coolwarm', interpolation = 'nearest', alpha=TRANSPARENCIA)


plt.savefig(os.path.join("heatmaps",f"{os.path.splitext(video_name)[0]}_coolwarm.png"))



