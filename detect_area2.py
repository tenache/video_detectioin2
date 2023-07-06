import cv2
import os
from time import sleep

parent_dir = "sample_data"
video = "camera_02-URQUIZA_main_20230703193142.dav"
video_path = os.path.join(parent_dir, video)

def get_rois(video_path):
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()

    rois = []  # List to hold all the ROIs

    print("Select your ROIs using the mouse. Press ENTER to confirm each ROI. Press ESC to stop selecting.")
    while True:
        roi = cv2.selectROI(frame)
        print(roi)
        # sleep(1)
        if sum([1 for coord in roi if coord> 0]) == 4:
            rois.append(roi)

        # Draw a rectangle around the ROI and show it
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        cv2.imshow("ROI Selection", frame)
        print(rois)
        # sleep(1)

        # Wait for ESC key to exit
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC key
            break
        

    cap.release()
    cv2.destroyAllWindows()

    for i, roi in enumerate(rois):
        print(f"roi {i+1} is {roi}")
        print(f"roi {i+1} type is {type(roi)}")
        # sleep(5)
        
    print(rois)
    # sleep(2)
    return rois

if __name__ == "__main__":
    get_rois(video_path)
    