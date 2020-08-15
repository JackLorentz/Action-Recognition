import argparse
import time
import cv2
import sys
import numpy as np

import torch.nn.parallel
import torch.optim
import torchvision

from my_dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale

GOP_SIZE = 12
SEG_SIZE = 40# ffmpeg mpeg4 rawvideo 5秒共有124幀


def cal_complementary_frames(frame_cnt, total_frame):
    print("Before compensation, original frames: ", frame_cnt)
    copy_list = []
    #掉幀率: 缺幀數 / 實際幀數
    if frame_cnt <= total_frame:
        rate = ((total_frame+1) - frame_cnt) / frame_cnt
    else:
        rate = (frame_cnt - (total_frame-1)) / frame_cnt
    #
    copy = False
    real_fcnt = 1
    comp_fcnt = 0
    while True:
        #決定該幀是否為補幀
        #攝影機非補幀
        if real_fcnt + comp_fcnt > total_frame:
            break
        if real_fcnt == 1 or real_fcnt*rate - comp_fcnt <= 1:
            copy = False
            real_fcnt += 1
        #攝影機補幀
        else:
            copy = True
            comp_fcnt += 1
        copy_list.append(copy)
    print("After compensation, Total frames: ", len(copy_list))

    return copy_list


def cal_complementary_segments(action_list, total_segment):
    print("Before compensation, original segments: ", len(action_list))
    result = []
    fcnt = len(action_list)
    rate = 0
    if len(action_list) <= total_segment:
        rate = ((total_segment+1) - fcnt) / fcnt
    else:
        rate = (fcnt - (total_segment-1)) / fcnt
    #決定該幀是否為補幀
    real_fcnt = 1
    comp_fcnt = 0
    while True:
        if real_fcnt + comp_fcnt > total_segment:
            break
        #非補幀
        if real_fcnt == 1 or real_fcnt*rate - comp_fcnt <= 1:
            result.append(action_list[real_fcnt-1])
            real_fcnt += 1
        #補幀
        else:
            if fcnt <= total_segment:
                result.append(action_list[real_fcnt-1])
            #當實際幀數比預期幀數多的時候,透過平均跳過的方式把多的幀數丟掉
            else:
                real_fcnt += 2
            comp_fcnt += 1
    print("After compensation, Total frames: ", len(result))

    return result


def actionRecongnize(args, net, cropping):
    #將影片存入資料結構, 並計算幀數找出哪些幀是iframe, mv和residual
    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            "input_data",
            "Action Recognition",
            video_name=args.video_name,
            representation=args.representation,
            transform=cropping,
            accumulate=(not args.no_accumulation)
            ),
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

    def forward_video(data):
        scores = net(data)
        scores = scores.view((-1, args.num_segments * 10) + scores.size()[1:])
        scores = torch.mean(scores, dim=1)
        return scores.data.cpu().numpy().copy()

    #辨識
    output = []
    proc_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            video_score = forward_video(data)
            output.append(video_score)

    cnt_time = time.time() - proc_start_time
    print('1 video done, average {:.1f} sec/video'.format(float(cnt_time)))
    
    return [np.argmax(x) for x in output], output


def main():
    parser = argparse.ArgumentParser(description="Action Recognition with Coviar (CVPR2018)")
    parser.add_argument('--video-name', type=str, default=None)
    parser.add_argument('--orig-video', type=str, default=None)
    parser.add_argument('--num-class', type=int, default=5)
    parser.add_argument('--num-segments', type=int, default=3)
    parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual'], default=None)
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--no-accumulation', action='store_true', help='disable accumulation of motion vectors and residuals.')
    args = parser.parse_args()

    #載入model
    net = Model(args.num_class, args.num_segments, args.representation, base_model=args.arch)
    checkpoint = torch.load(args.model)
    
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    #data argumentaion預設每個segment產生10個
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
    ])

    #我們只有一個GPU, 所以只能設cudo(0) => GPU0
    devices = [0]
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    #姿態辨識
    result, scores = actionRecongnize(args, net, cropping)
    np.savez(args.representation + "_recognizer_score.npz", scores=np.array(scores), labels=np.array(result))
    #print(len(result))
    
    #讀檔
    cap = cv2.VideoCapture(args.orig_video)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        return
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    copy_list = cal_complementary_frames(frame_num, 1500)
    action_list = cal_complementary_segments(result, 38)

    #寫檔
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./result.avi', fourcc, 30.0, (1920, 1080))

    #產生影片
    frame_cnt = 0
    frame = np.array([])
    for i in range(len(copy_list)):
        #
        if not copy_list[i]:
            _, frame = cap.read()
        #
        else:
            #當實際幀數比預期幀數多的時候,透過平均跳過的方式把多的幀數丟掉
            if frame_num > 1500:
                _, frame = cap.read()
                _, frame = cap.read()
        #
        if action_list[(i // SEG_SIZE)] == 1:
            cv2.putText(frame, "WritingOnBoard", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Non-WritingOnBoard", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        #
        out.write(frame)
        frame_cnt += 1
        print('\rcurrent/total frame = {}/{} / Progress: {:.1f}%'.format(frame_cnt, frame_num, (frame_cnt / frame_num)*100), end='')
        sys.stdout.flush()
    out.release()
    cap.release()


if __name__ == '__main__':
    main() 