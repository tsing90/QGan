import numpy as np
import torch
from tqdm import tqdm
import sys, os
from pathlib import Path
import cv2
from pose_transform import translate

# ## make label images for pix2pix

def generate(origin_img, img_dir, label_dir, size_dst, size_crop, crop_from, pose_transform=False):
    # Pose estimation (OpenPose)
    openpose_dir = Path('../src/pytorch_Realtime_Multi-Person_Pose_Estimation/')

    sys.path.append(str(openpose_dir))
    sys.path.append('../src/utils')
    # from Pose estimation
    from evaluate.coco_eval import get_multiplier, get_outputs
    # utils
    from openpose_utils import remove_noise, get_pose, get_pose_coord, get_pose_new


    model = pose_model()

    total = len(list(origin_img.iterdir()))
    img_idx = range(total)

    if pose_transform:
        ratio_src, ratio_tar = '../data/source/ratio_a.png', '../data/target/ratio_b.png'
        if not os.path.isfile(ratio_src):
            raise TypeError('Directory not exists: {}'.format(ratio_src))
        if not os.path.isfile(ratio_tar):
            raise TypeError('Directory not exists: {}'.format(ratio_tar))

        imgset = [ratio_src, ratio_tar]
        origin = []
        height = []
        ratio = {'0-1': None, '1-2': None, '2-3': None, '3-4': None, '1-8': None, '8-9': None,
                 '9-10': None, '0-14': None, '14-16': None}  # target/source
        coord = {'0-1': [], '1-2': [], '2-3': [], '3-4': [], '1-8': [], '8-9': [], '9-10': [], '0-14':[], '14-16':[]}  # len of joint
        # co_tar = {'0-1':None, '1-2':None, '2-3':None,'3-4':None,'1-8':None,'8-9':None,'9-10':None}

        for img_path in imgset:
            img = cv2.imread(str(img_path))
            if not img.shape[:2] == size_dst[::-1]:  # format: (h, w)
                img = img_resize(img, size_crop, crop_from, size_dst)  # size_dst format: (W, H)
            multiplier = get_multiplier(img)
            with torch.no_grad():
                paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
            r_heatmap = np.array([remove_noise(ht)
                                  for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
                .transpose(1, 2, 0)
            heatmap[:, :, :-1] = r_heatmap
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}  # only 'thre2' matters

            label, joint_list = get_pose_coord(img, param, heatmap, paf)
            # print ('joint list: \n',joint_list)

            origin.append(joint_list[1][0][:2])  # we set the no.1 pose (neck) as the original ref. point
            height_max = max(joint_list, key=lambda x: x[0][1])[0][1]
            height_min = min(joint_list, key=lambda x: x[0][1])[0][1]
            height.append(height_max - height_min)

            for k in ratio.keys():
                klist = k.split('-')
                j_1, j_2 = int(klist[0]), int(klist[-1])
                # assert j_1 == int(joint_list[j_1][0][-1]) and j_2 == int(
                #    joint_list[j_2][0][-1])  # may cause issue if empty array exists
                co_1, co_2 = list(joint_list[j_1][0][:2]), list(joint_list[j_2][0][:2])
                j_len = ((co_1[0] - co_2[0]) ** 2 + (co_1[1] - co_2[1]) ** 2) ** 0.5
                coord[k].append(j_len)

        for k, v in coord.items():
            src_len, tar_len = v[0], v[1]
            ratio[k] = tar_len / src_len

        ratio_body = height[1] / height[0]  # target / source height
        print('ratio:\n', ratio, '\nratio_body:', ratio_body)  # test only

    for idx in tqdm(img_idx):
        img_path = origin_img.joinpath('img_{:04d}.png'.format(idx))
        img = cv2.imread(str(img_path))

        if not img.shape[:2] == size_dst[::-1]:
            # set crop size and resize
            img = img_resize(img, size_crop, crop_from, size_dst)  # size format: (W, H)

        multiplier = get_multiplier(img)
        with torch.no_grad():
            paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
        r_heatmap = np.array([remove_noise(ht)
                              for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
            .transpose(1, 2, 0)
        heatmap[:, :, :-1] = r_heatmap
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}  # only thre2 makes effect

        if pose_transform:
            _, joint_list = get_pose_coord(img, param, heatmap, paf)
            #print('joint_list', '\n', joint_list)  # test only
            new_joint = translate(joint_list, ratio, origin, ratio_body)
            new_joint_list = new_joint.run()
            #print('joint_list new', '\n', new_joint_list)  # test only
            """
            with open('joint_list.txt','a') as f:
                f.write('joint_list_{}\n'.format(idx)+str(joint_list)+'\nnew_joint_list_{}\n'.format(idx)+str(new_joint_list)+'\n')
            """
            label = get_pose_new(img, param, heatmap, paf, new_joint_list)
        else:
            label = get_pose(img, param, heatmap, paf)  # size changed !!!

        cv2.imwrite(str(img_dir.joinpath('img_{:04d}.png'.format(idx))), img)
        cv2.imwrite(str(label_dir.joinpath('label_{:04d}.png'.format(idx))), label)

    torch.cuda.empty_cache()  #
    print(str(total) + ' ' + str(origin_img.parent.name) + ' images are generated')


def pose_model():
    # Pose estimation (OpenPose)
    openpose_dir = Path('../src/pytorch_Realtime_Multi-Person_Pose_Estimation/')

    sys.path.append(str(openpose_dir))

    # get_ipython().run_line_magic('load_ext', 'autoreload')
    # get_ipython().run_line_magic('autoreload', '2')

    # openpose
    from network.rtpose_vgg import get_model

    weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')
    # weight_name.mkdir(exist_ok=True)

    model = get_model('vgg19')
    model.load_state_dict(torch.load(str(weight_name)))
    model = torch.nn.DataParallel(model).cuda()
    # model.float()
    # model.eval()

    return model

def img_resize(img, crop_size, crop_from, dst_size):

    # img: (Height, Width); crop_size: (Width, Height)

    if not img.shape[0]/img.shape[1] == dst_size[1]/dst_size[0]:
        # padding
        if crop_size[0]>img.shape[1] or crop_size[1]>img.shape[0]:

            if crop_size[0]>img.shape[1]:
                pad_width = (crop_size[0] - img.shape[1])//2
                if crop_from == 'central':
                    img = cv2.copyMakeBorder(img, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT)
                elif crop_from == 'left':
                    img = cv2.copyMakeBorder(img, 0, 0, 2*pad_width, 0, cv2.BORDER_CONSTANT)
            elif crop_size[1] > img.shape[0]:
                pad_height = (crop_size[1] - img.shape[0])//2
                if crop_from == 'central':
                    img = cv2.copyMakeBorder(img, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT)
                elif crop_from == 'top':
                    img = cv2.copyMakeBorder(img, 2*pad_height, 0, 0, 0, cv2.BORDER_CONSTANT)

        # crop
        else:
            if crop_from == 'central':
            # crop | crop_size = (width, height)
                oh = (img.shape[0] - crop_size[1]) // 2
                ow = (img.shape[1] - crop_size[0]) // 2
                img = img[oh:oh + crop_size[1], ow:ow + crop_size[0]]
            elif crop_from == 'top':
                oh = img.shape[0] - crop_size[1]
                ow = (img.shape[1] - crop_size[0]) // 2
                img = img[oh:oh + crop_size[1], ow:ow + crop_size[0]]

    # resize | dst_size = (width, height)
    img = cv2.resize(img, dst_size)  # resolution setting

    return img


if __name__ == "__main__":
    source_dir = Path('../data/source')
    source_img_dir = Path('../data/source/images')
    source_img_dir.mkdir(exist_ok=True)
    test_img_dir = source_dir.joinpath('test_img')
    test_img_dir.mkdir(exist_ok=True)
    test_label_dir = source_dir.joinpath('test_label')
    test_label_dir.mkdir(exist_ok=True)

    size_dst = (512, 288)
    size_crop = (640, 360)
    crop_from = 'central'
    pose_transform = True

    generate(source_img_dir, test_img_dir, test_label_dir, size_dst, size_crop, crop_from, pose_transform)

