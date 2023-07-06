
import cv2
import numpy as np
import os
from tkinter import *
from PIL import Image, ImageTk


parent_dir = "sample_data"
video = "camera_02-URQUIZA_main_20230703193142.dav"
video_path = os.path.join(parent_dir, video)

# Load video
cap = cv2.VideoCapture(video_path)




class LineDrawer:
    def __init__(self, canvas, cap):
        self.points = []
        self.canvas = canvas
        self.cap = cap
        
    def click(self, event):
        while len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.draw_line()
        
            

            
        
    def draw_line(self):
        # Define boundary

        while(self.cap.isOpened()):
            # Read frame
            ret, frame = self.cap.read()

            if ret == True:
                # Classify areas
                mask = np.zeros_like(frame)
                mask[y,x] = [255, 0, 0]
                within_area = True
                for boundary_points in self.points:
                    for y in range(frame.shape[0]):
                        for x in range(frame.shape[1]):
                            if not self.is_in_area((x, y), boundary_points):
                                within_area = False
                                break
                    if within_area:
                        mask[y, x] = [0, 255, 0] # paint inside agreen
                    else:
                        mask[y,x] = [255, 0, 0] # paint the outside red

                

                # Convert the OpenCV image (BGR) to a PIL image (RGB) and then to ImageTk
                alpha = 0.3
                frame = cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(im)

                # Update the image in the canvas
                self.canvas.create_image(0, 0, anchor = NW, image = img)

                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    break
            else:
                break
        
        
    def is_in_area(self, point, boundary_points):
        (x1, y1), (x2, y2) = boundary_points
        x, y = point

        # Calculate cross product
        value = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

        # If value is positive, the point is on one side of the line
        # If value is negative, the point is on the other side
        return value > 0

            
        
        

        