import numpy as np
import cv2, argparse, sys

# 需要在720P下才能做辨識
source_width = 1920
source_height = 1080
goal_width = 1280
goal_height = 720

ROI_X = int(1425*goal_width/source_width)
ROI_Y = int(531*goal_height/source_height)
ROI_WIDTH = 340
ROI_HEIGHT = 256

if ROI_X + ROI_WIDTH > 1280:
    ROI_X = ROI_X - (ROI_X+ROI_WIDTH-1280)

if ROI_Y + ROI_HEIGHT > 720:
    ROI_Y = ROI_Y - (ROI_Y+ROI_HEIGHT-720)

parser = argparse.ArgumentParser(description="Crop")
parser.add_argument('--video-name', type=str, default=None)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_name)
if not cap.isOpened():
    print("Open failed.")
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#寫檔
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.video_name[:-4] + '_closeUp_in_720P.avi', fourcc, 30.0, (ROI_WIDTH, ROI_HEIGHT))

frame_cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
        roi = frame[int(ROI_Y):int(ROI_Y+ROI_HEIGHT), int(ROI_X):int(ROI_X+ROI_WIDTH)]
        cv2.imshow('Display', roi)
        cv2.waitKey(10)
        out.write(roi)
        frame_cnt += 1
        print('\rcurrent/total frame = {}/{} / Progress: {:.1f}%'.format(frame_cnt, frame_num, (frame_cnt / frame_num)*100), end='')
        sys.stdout.flush()
    else:
        break
out.release()
cap.release()