
import img_process, pose_generate, train, make_mp4
from pathlib import Path

## setting

# cut video
src_cut_video = False  # whether cut video
src_cut_start = 5  # in seconds
src_cut_end = 25
src_name = 'source.mp4'
tar_name = 'target.mp4'

# image reshape: ATTENTION: make sure final shape is divisible by 32 or 16
# all size are in format (width, height)
src_size_dst = (512, 288)  # final shape after scaling with fixed aspect ratio
src_size_crop = (640, 360)  # intermediate size after crop/padding
src_crop_from = 'central'  # how to crop images

tar_size_dst = (512, 288)
tar_size_crop = (960, 540)
tar_crop_from = 'central'

# training and inferencing
loadSize = 512  # load image's width, we set it equal to fineSize (after crop and scale)
continue_train = True  # load saved train model parameters, resume training
inference_only = False  # if true: without train, do inference ONLY !

# output video
mv_style = 'two'  # choose how results are displayed: one - generated target only; two - src + generated target;



## source pre-processing
source_dir = Path('../data/source/')
source_dir.mkdir(exist_ok=True)

source_img_dir = source_dir.joinpath('images')
source_img_dir.mkdir(exist_ok=True)

#img_process.download(source_dir)  # download source video
if src_cut_video:
    src_output_dir = str(source_dir)+'/source_'+str(src_cut_start)+'_'+str(src_cut_end)+'.mp4'
    src_name = 'source_'+str(src_cut_start)+'_'+str(src_cut_end)+'.mp4'
    img_process.cutmv(source_dir.joinpath(src_name), src_output_dir, src_cut_start, src_cut_end)

img_process.mv2img(source_dir, source_img_dir, src_name)  # transform video to images

## target pre-processing
target_dir = Path('../data/target/')
target_dir.mkdir(exist_ok=True)

target_img_dir = target_dir.joinpath('images')
target_img_dir.mkdir(exist_ok=True)

img_process.mv2img(target_dir, target_img_dir, tar_name)  #


## generate pose estimation label images
# - for source images

test_img_dir = source_dir.joinpath('test_img')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = source_dir.joinpath('test_label')
test_label_dir.mkdir(exist_ok=True)

if len(list(test_img_dir.iterdir()))!=0 and len(list(test_label_dir.iterdir()))!=0:
    print ('source labels were already generated!')
else:
    pose_transform = True  # need to be True
    pose_generate.generate(source_img_dir, test_img_dir, test_label_dir, \
                           src_size_dst, src_size_crop, src_crop_from, pose_transform)

# - for train images 

train_dir = target_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)

if len(list(train_img_dir.iterdir()))!=0 and len(list(train_label_dir.iterdir()))!=0:
    print ('target labels were already generated!')
else:
    pose_transform = False
    pose_generate.generate(target_img_dir, train_img_dir, train_label_dir, \
                           tar_size_dst, tar_size_crop, tar_crop_from, pose_transform)

# training target images
if not inference_only:
    if_train = True
    train_param_dir = '../data/train_opt.pkl'
    train.train_target(train_param_dir, if_train, loadSize, continue_train)

# dance transfer
if_train = False
transfer_param_dir = '../data/test_opt.pkl'
print('inferencing images ...')
train.train_target(transfer_param_dir, if_train, loadSize)

# make video
result_dir = Path('../results/target/test_latest/images')
result_dir.mkdir(exist_ok=True)
audio_dir = source_dir.joinpath(src_name)


print ('making video ...')
make_mp4.make(test_img_dir, result_dir, audio_dir, mv_style)
print ('finish !')




