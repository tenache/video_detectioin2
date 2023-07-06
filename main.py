from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet 
import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random
from detect_area2 import get_rois

from functools import reduce

video_name = "camera_01-FRENTE_main_20230706103637.dav"
video_path = os.path.join(".","sample_data",video_name)


output_dir = "out_data"
video_out_path = os.path.join( output_dir, os.path.splitext(video_name)[0] + "_out.mp4")

rois = get_rois(video_path)


def detect_frames(video_name=video_name, video_path=video_path,video_out_path =video_out_path, show_me=True, max_frames= None, rois=rois):

    # cap = cv2.VideoCapture("rtsp://thomas:thomas123456@172.20.208.1:554/cam/realmonitor?channel=1&subtype=0")
  
    cap = cv2.VideoCapture(video_path)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    all_people = set()
    all_people_roi = [set() for _ in range(len(rois))] 

    ret, frame = cap.read()
    print(f"frame is {frame}")

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]) )

    model = YOLO("yolov8n.pt")
    # model = YOLO("yolov3n.pt")

    tracker = Tracker()

    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(10)]
    all_detections = []
    all_detections_roi = [[] for _ in range(len(rois))]
    frame_count = 0
    if max_frames is None:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while ret and frame_count < max_frames:
    # while frame_count < 25:
        results = model(frame)
        result = results[0]
        detections = []
        detections_roi = [[] for _ in range(len(rois))]
        
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r # all objects in coco_classes
            if class_id == 0:
                detections.append([int(x1), int(y1),int(x2), int(y2), int(score)])
                all_detections.append((int(x1),int(y1),int(x2),int(y2)))
                if rois is not None:
                    for i, roi in enumerate(rois):
                        if roi[0] < x1 and x2 < roi[2] and roi[1] < y2 < roi[3]:
                            detections_roi[i].append([int(x1), int(y1),int(x2), int(y2), int(score)])
                            all_detections_roi[i].append((int(x1),int(y1),int(x2),int(y2)))
                            
                            
       
        # Definir una funcion "count people", o algo similar, para contar el numero de personas que pasan por las camaras ... 
        # por que definir una funcion count-people? porque necesitas que sea algo un poco mas sofisticado que simplemente  
        # contar el numero de detecciones o tracks....
        # A veces la camara se equivoca y cuenta dos veces a la misma persona, entonces necesitamos establecer algunas reglas basicas ...
        # Por ejemplo, la persona no puede "aparecer de la nada"
        # Empecemos por una funcion ultra basica, que cuente todo... y de ahi vamos haciendo cosas mejores ... 
        # print(f"detections is {detections}")
        if detections:
            tracker.update(frame, detections)  # the most importante function 
            for track in tracker.tracks:
                all_people.add(track.track_id)
                
        if rois is not None:
            for i, roi in enumerate (rois):
                if detections_roi[i]:
                    print(rois)
                    tracker.update(frame, detections_roi[i])
                    for track in tracker.tracks:
                        all_people_roi[i].add(track.track_id)
       
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)  # Text color (in BGR format)
        thickness = 2  # Text thickness
        # text = "be happy"
        text = f"{len(reduce(set.union, all_people_roi))} personas de {len(all_people)} entraron "
        print("all people roi is ")
        print(set.union, all_people_roi)
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

            
        # Specify the position to write the text
        position = (int((video_width - text_width) / 2), video_height - int(0.05 * video_height))               
        
        if tracker.tracks is not None:
            print(f"len of tracker.tracks is {len(all_people)}")
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
                if show_me:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
                    # cv2.rectangle(frame,((x1, y1),(x2, y2), (colors[track_id % len(colors)])),3)
            
                
        # print(detections)
        if show_me:    
            cv2.imshow("frame", frame)
            cv2.waitKey(10)
        cap_out.write(frame) # metemos el frame en el output file
        ret, frame = cap.read() # leemos nuevos frame
        frame_count +=1
        

    
    cap.release() # no estoy seguro que hace, pero el vago parecia asustado cuando se olvido...
    cap_out.release()  # no estoy seguro que hace, pero el vago parecia asustado cuando se olvido...
    cv2.destroyAllWindows() # hahahaha!
    print(f"frame count is {frame_count}")
    return all_detections, video_height, video_width 

if __name__ == "__main__":
    detect_frames()
    