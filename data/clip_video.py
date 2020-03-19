from moviepy.editor import *
import os

def clip(video_address):
    dirs = os.path.dirname(video_address)
    video = VideoFileClip(video_address)
    duration = video.duration
    batch = int(duration//10)
    print("the number of batches of this video is {batchnum}".format(batchnum = batch))
    video.close()
    del video
    
    for i in range(batch):
        video = VideoFileClip(video_address)
        clip = video.subclip(i*10, (i+1)*10)
        new_add = dirs + '/'+ str(i) +'.mp4'
        clip.write_videofile(new_add)
        clip.close()
        del clip
        video.close()
        del video
        
    print("all work done")
    
    
if __name__ == '__main__':
    video_address = ''
    clip(video_address)
