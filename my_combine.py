"""Combine testing results of the three models to get final accuracy."""

import argparse
import time
import cv2
import sys
import numpy as np

from coviar import get_num_frames

GOP_SIZE = 12
SEG_SIZE = 41

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
            if frame_cnt <= total_frame:
                copy = True
            #當實際幀數比預期幀數多的時候,透過平均跳過的方式把多的幀數丟掉
            else:
                real_fcnt += 2
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

def main():
    parser = argparse.ArgumentParser(description="combine predictions")
    parser.add_argument('--video-name', type=str, default=None)
    parser.add_argument('--iframe', type=str, required=True,
                        help='iframe score file.')
    parser.add_argument('--mv', type=str, required=True,
                        help='motion vector score file.')
    parser.add_argument('--res', type=str, required=True,
                        help='residual score file.')

    parser.add_argument('--wi', type=float, default=1.0,
                        help='iframe weight.')
    parser.add_argument('--wm', type=float, default=1.0,
                        help='motion vector weight.')
    parser.add_argument('--wr', type=float, default=1.0,
                        help='residual weight.')

    args = parser.parse_args()

    iframe = np.load(args.iframe, allow_pickle=True)
    mv = np.load(args.mv, allow_pickle=True)
    residual = np.load(args.res, allow_pickle=True)

    i_score = iframe['scores']
    mv_score = mv['scores']
    res_score = residual['scores']

    #三個模型綜合評分
    combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
    combined_score = combined_score.reshape(combined_score.shape[0], combined_score.shape[2])
    results = np.argmax(combined_score, axis=1)

    #讀檔
    cap = cv2.VideoCapture(args.video_name)
    if not cap.isOpened():
        print('Error opening video stream or file')
        return
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    copy_list = cal_complementary_frames(total_frame, 1800)
    action_list = cal_complementary_segments(results, 45)

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
            if total_frame > 1800:
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
        print('\rcurrent/total frame = {}/{} / Progress: {:.1f}%'.format(frame_cnt, total_frame, (frame_cnt / total_frame)*100), end='')
        sys.stdout.flush()
    out.release()
    cap.release()

if __name__ == '__main__':
    main()
