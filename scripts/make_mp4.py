from pathlib import Path
from moviepy.editor import *
import glob
import cv2

# make a video from sequtial images
# two ways to import img:
#1. folder (all img inside should be in alpha order)
#2. file list (ordered)

def make(source_dir, target_dir, audio_dir, mv_style):

    shape = cv2.imread(str(next(source_dir.iterdir()))).shape

    if mv_style == 'one':
        img_seq_t = sorted(glob.glob(str(target_dir.joinpath('*synthesized*'))))
        new = ImageSequenceClip(img_seq_t, fps=24)
        new_name = 'output_one.mp4'

    elif mv_style == 'two':

        # if put video in sequence, use 'concatenate_videoclips([clip1,clip2,clip3])'
        # if put video in the same page, use following:
        # new = CompositeVideoClip([img_seq_t.set_pos((0,0)),
        #                          img_seq_t.set_pos((490,0))],
        #                          size = (970,272))

        img_seq_s = ImageSequenceClip(str(source_dir), fps=24)  # path should be string
        img_seq_t = sorted(glob.glob(str(target_dir.joinpath('*synthesized*'))))
        img_seq_t = ImageSequenceClip(img_seq_t, fps=24)
        new = CompositeVideoClip([img_seq_s.set_pos((0,0)),
                                  img_seq_t.set_pos((shape[1]+10,0))],
                                  size = (shape[1]*2+10, shape[0]))
        new_name = 'output_two.mp4'

    elif mv_style == "three":
        img_seq_s = ImageSequenceClip(str(source_dir), fps=24)  # path should be string
        img_seq_mid = sorted(glob.glob(str(target_dir.joinpath('*input*'))))
        img_seq_mid = ImageSequenceClip(img_seq_mid, fps=24)
        img_seq_t = sorted(glob.glob(str(target_dir.joinpath('*synthesized*'))))
        img_seq_t = ImageSequenceClip(img_seq_t, fps=24)
        new = CompositeVideoClip([img_seq_s.set_pos((0,0)),
                                  img_seq_mid.set_pos((shape[1]+10,0)),
                                  img_seq_t.set_pos((shape[1]*2+20,0))],
                                  size = (shape[1]*3+20, shape[0]))
        new_name = 'output_three.mp4'

    # add audio
    audio = AudioFileClip(str(audio_dir))
    new = new.set_audio(audio)
    new = new.volumex(1.2)

    new.write_videofile(new_name, audio_codec='aac')
    
if __name__ == "__main__":
    
    test_img_dir = Path('../data/source/test_img')
    result_dir = Path('../results/target/test_latest/images')
    result_dir.mkdir(exist_ok=True)
    audio_dir = Path('../data/source/source_5_25.mp4')
    mv_style = 'one'

    make(test_img_dir, result_dir, audio_dir, mv_style)
