"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import cv2
import os.path
import random
import torchvision

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load

from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from transforms import color_aug

'''
影片前處理: 必須是rawvideo mpeg4檔(720P), 裁下ROI的區域後再縮放長寬比為340 : 256
'''

GOP_SIZE = 12
SEG_SIZE = 41# ffmpeg mpeg4 rawvideo 5秒共有124幀

# 需要在720P下才能做辨識
source_width = 1920
source_height = 1080
goal_width = 1280
goal_height = 720

#test2.avi: ROI left up corner: (x, y) = (234, 270), width x height: 340 x 256
ROI_X = int(1425*goal_width/source_width)
ROI_Y = int(531*goal_height/source_height)
ROI_WIDTH = 340
ROI_HEIGHT = 256

#超出範圍要修正
if ROI_X + ROI_WIDTH > 1280:
    ROI_X = ROI_X - (ROI_X+ROI_WIDTH-1280)

if ROI_Y + ROI_HEIGHT > 720:
    ROI_Y = ROI_Y - (ROI_Y+ROI_HEIGHT-720)


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_gop_pos(frame_idx, representation):
    gop_idx = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_idx -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_idx, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_name,
                 representation,
                 transform,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._representation = representation
        self._transform = transform
        self._accumulate = accumulate
        self._frames = []

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_video(video_name)


    def _load_video(self, video_name):
        #選擇擷取特徵
        representation_idx = 0
        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2

        #計算片段數
        total_frames = get_num_frames(video_name)
        total_segments = total_frames // SEG_SIZE

        #把每個片段中間那幀紀錄下來
        frames = []
        for i in range(total_segments):
            gop_idx, gop_pos = self._get_frame_index(total_frames, i)
            img = load(video_name, gop_idx, gop_pos, representation_idx, self._accumulate)
            roi_img = img[int(ROI_Y):int(ROI_Y+ROI_HEIGHT), int(ROI_X):int(ROI_X+ROI_WIDTH)]
            frames.append(roi_img)

        #預設是每3個片段辨識一個動作
        for i in range(2, len(frames)):
            tmp = []
            tmp.append(frames[i-2])
            tmp.append(frames[i-1])
            tmp.append(frames[i])
            self._frames.append(tmp)
        frames.clear()


    def _get_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1
        
        v_frame_idx = int(np.round(SEG_SIZE * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)


    def __getitem__(self, index): 

        frames = []
        tmp = self._frames[index]
        for _, img in enumerate(tmp):
            if img is None:
                print('Error: loading video failed.')
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            else:
                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                img = color_aug(img)
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img = img[..., ::-1]

            frames.append(img)

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input

    def __len__(self):
        return len(self._frames)