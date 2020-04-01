# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

def video2frame_at2FPS(name):
    add = './videos/' + name
    cap = cv2.VideoCapture(add)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    output_dir = './frames/' + str(name[:-4])
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print ('Error')
        
    currentFrame = 0.0 #if FPS changes then the corresponding count number shall also change 
    while currentFrame < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (currentFrame*500)) #currentFrame*500 == FPS@2, currentFrame*1000 == FPS@1, ... etc.
        ret, frame = cap.read()
        
        title = output_dir + '/' + '_frame' + str(int(currentFrame*2)) + '.jpg' # #if FPS changes then the corresponding count number shall also change
        print('creating... '+ title)
        cv2.imwrite(title, frame)
        
        currentFrame += 0.5 #if FPS changes then the corresponding count number shall also change
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    base_dir = './videos'
    v_list = os.listdir(base_dir)
    for item in v_list:
        video2frame_at2FPS(item)
