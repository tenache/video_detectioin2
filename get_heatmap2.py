from main import detect_frames
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from time import sleep
import pickle as pkl
import seaborn as sns
from numpy.ma import masked_array
from detect_area2 import get_roi

OPACIDAD = 0.4
define_region = True
video_name = "camera_02-URQUIZA_main_20230703193142.dav"
video_path = os.path.join(".","sample_data",video_name)
video_out_path = os.path.join(".","out_data", video_name + "_out.mp4")
region_or_line = "line"
if define_region:
    roi = get_roi(video_path, region_or_line)
else:
    roi = None



# tomar detecciones
all_detections, height, width = detect_frames(video_name, video_path, video_out_path, show_me=False, max_frames=None, roi=roi)

# vamos a hacer un mapa de calor 
heatmap = np.zeros((height, width))

# esto lo sacamos simplemente para tener una imagen sobre la cual dibujar el mapa de calor
cap = cv2.VideoCapture(video_path)
ret,frame = cap.read(0)

# Esta funcion es para ayudar a suavizar la imagen, nor hacerla tan con partes solidas ... 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

for detection in all_detections:
    print(detection)
    x1, y1, x2, y2 = detection
    width, height = x2-x1, y2-y1

    # Hacemos el "Gausian kernel" para suavizar
    gk = gaussian_kernel(max(width, height), sigma=min(width, height)/2.0)
    
    # ajustamos el tamanio del kernel al tamanio del bb
    gk = cv2.resize(gk, (width, height))
    
    # Get the center coordinates of the detection
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Determine the top left corner of the kernel placement
    kx1, ky1 = cx - gk.shape[1] // 2, cy - gk.shape[0] // 2
    
    # nos aseugramos que el mapa de calor que vamos a agregar al kernel sea al menos tan grande como el kernel
    # Add the kernel to the heatmap, ensuring we don't go out of bounds
    for i in range(gk.shape[0]):
        for j in range(gk.shape[1]):
            if ky1+i >= 0 and ky1+i < heatmap.shape[0] and kx1+j >= 0 and kx1+j < heatmap.shape[1]:
                heatmap[ky1+i, kx1+j] += gk[i, j]

    
# Normalizamos

print(f"np.max(heatmap is {np.max(heatmap)}")
print(f"heatmap {heatmap}")

range_heatmap = np.max(heatmap) - np.min(heatmap)
minimum = range_heatmap/20
heatmap = (heatmap - np.min(heatmap)) / range_heatmap
heatmap_masked = masked_array(heatmap, heatmap < minimum)

# Desplegamos el heatmap
print(f"heatmap is {heatmap}")
with open(os.path.splitext(video_name)[0] + "pickle",'wb') as handle:
    pkl.dump(heatmap_masked, handle)


plt.imshow(frame)
plt.imshow(heatmap_masked, cmap = 'bwr', interpolation = 'nearest', alpha=OPACIDAD)
plt.savefig(os.path.join("heatmaps",f"{os.path.splitext(video_name)[0]}_bwr.png"))
plt.imshow(heatmap_masked, cmap = 'coolwarm', interpolation = 'nearest', alpha=OPACIDAD)


plt.savefig(os.path.join("heatmaps",f"{os.path.splitext(video_name)[0]}_coolwarm.png"))

