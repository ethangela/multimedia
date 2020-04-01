# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from commons.executions import multiple_executions_wrapper

SEGMENTED_CLIPS_ROOT    = os.environ["SEGMENTED_CLIPS_ROOT"]
IMAGE_ROOT              = os.environ["IMAGE_ROOT"]
SAMPLE_FPS              = int(os.environ.get("SAMPLE_FPS", -1))

@multiple_executions_wrapper
def video2frame_at2FPS(name):
    cap = cv2.VideoCapture(name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    video_name  = name.split("/")[-1].replace(".mp4", "")
    output_dir  = os.path.join(IMAGE_ROOT, video_name.replace(".mp4", "")) 
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print ('Error')
    
    if SAMPLE_FPS > 0:
        currentFrame = 0.0 #if FPS changes then the corresponding count number shall also change 
        while currentFrame < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, (currentFrame*500)) #currentFrame*500 == FPS@2, currentFrame*1000 == FPS@1, ... etc.
            ret, frame = cap.read()
            
            title   = os.path.join(output_dir, "{0}_fps_{1}.jpg".format(video_name, str(currentFrame*2)))
            print('creating... '+ title)
            cv2.imwrite(title, frame)
            
            currentFrame += 0.5 #if FPS changes then the corresponding count number shall also change
    else:
        # Do not subsample by FPS
        success     = cap.grab()
        frame_indx  = 0
        while success:
            _, frame = cap.retrieve()
            
            title   = os.path.join(output_dir, "{0}_fps_{1}.jpg".format(video_name, frame_indx))
            print('creating... '+ title)
            cv2.imwrite(title, frame)
            frame_indx += 1
            success = cap.grab()   
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    base_dir = '/Users/johan.kok/Downloads/multimedia/subclips/Belly_dance/'

    for directory, sub_dir, files in os.walk(SEGMENTED_CLIPS_ROOT):
        for each_file in files:
            if ".MP4" in each_file.upper():
                video_path = os.path.join(directory, each_file)
                video2frame_at2FPS(name = video_path)
