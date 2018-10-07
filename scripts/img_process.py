import cv2
from pytube import YouTube
from pathlib import Path
from moviepy.editor import *

def download(save_dir, fname):

    # download: Bruno Mars - That's What I Like
    #fname = 'source'
    if not save_dir.joinpath(fname+'.mp4').exists():
        print ('Download {} ......'.format(fname))
        #yt = YouTube('https://www.youtube.com/watch?v=PMivT7MJ41M')
        yt = YouTube('https://www.youtube.com/watch?v=PMivT7MJ41M')
        yt.streams.first().download(str(save_dir), fname)
    else:
        print (fname+'.mp4', ' already downloaded')

def cutmv(save_dir, new, start, end):
    video_s = VideoFileClip(str(save_dir)).subclip(start, end)
    video_s.write_videofile(new)

def mv2img(save_dir, img_dir, name, max = 10000):
    
    if len(list(img_dir.iterdir()))!=0:
        print (str(name)+' images were already generated!')
    
    else:
        cap = cv2.VideoCapture(str(save_dir.joinpath(name)))
        i = 0
        while(cap.isOpened()):
            flag, frame = cap.read()
            if flag == False or i == max:  # max generated: 10000
                break
            cv2.imwrite(str(img_dir.joinpath('img_{:04d}.png'.format(i))), frame)
            i += 1
        print (str(i)+' '+str(name)[:-4]+' images are generated')
        
        
if __name__ == "__main__":

    save_dir = Path('../data/source/')
    save_dir.mkdir(exist_ok=True)

    img_dir = save_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)
    
    name = 'source.mp4'
    start = 5
    end = 25
    output_dir = str(save_dir)+'/source_'+str(start)+'_'+str(end)+'.mp4'
    new_name = 'source_'+str(start)+'_'+str(end)+'.mp4'
    if_download = False

    if if_download:
        print('downloading video ...')
        download(save_dir, name)

    #cutmv(save_dir.joinpath(name), output_dir, start, end)
    #mv2img(save_dir, img_dir, 'source_5_25.mp4')
    
    
