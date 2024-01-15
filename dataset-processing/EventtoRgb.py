import os 
import pdb 
import csv 
import numpy as np 
import cv2
import torch
# import pandas as pd
from tqdm import tqdm
# import aedat
import pdb
from dv import AedatFile
import scipy.io as sio


data_path = r"/''''''/Poker_Event/"
save_path = r"/''''''/Poker_rgbframes/"
video_files = os.listdir(data_path)
if __name__ == '__main__':
	device = torch.device("cuda:0")
	for fileID in range(len(video_files)):
		videoName = video_files[fileID]
		fileLIST = os.listdir(data_path + videoName + r'/')
		save_csv_path_img = save_path + videoName + r'/'

		if not os.path.exists(save_csv_path_img):
			os.makedirs(save_csv_path_img)
		for newFileID in tqdm(range(len(fileLIST))):
			csv_videoName = fileLIST[newFileID]
			rgb_save_path_img_dvs = save_csv_path_img +csv_videoName[:-7]+'/'
			if os.path.exists(rgb_save_path_img_dvs):
				continue
			if not os.path.exists(rgb_save_path_img_dvs):
				os.makedirs(rgb_save_path_img_dvs)
			aedat_file_path = data_path + videoName + r'/' + csv_videoName
	
			frame_all = []
			frame_exposure_time = []
			# frame_interval_time = []
			with AedatFile(aedat_file_path) as f:
				# list all the names of streams in the file
				# print(f.names)
				# extract timestamps of each frame 
				for frame in f['frames']:
					frame_all.append(frame.image)   
					frame_exposure_time.append([frame.timestamp_start_of_exposure, frame.timestamp_end_of_exposure])   
					# frame_interval_time.append([frame.timestamp_start_of_frame,    frame.timestamp_end_of_frame])
			
				frame_timestamp = frame_all
			# elif use_mode == 'frame_interval_time':
			#     frame_timestamp = frame_interval_time

			frame_num = len(frame_timestamp)

			height, width = 260,346
			for frame_no in range(0, int(frame_num)-1):
				this_frame = frame_all[frame_no]
				event_img = np.zeros((height, width))
				cv2.imwrite(os.path.join(rgb_save_path_img_dvs, '{:04d}'.format(frame_no)+'.png'), this_frame)
			
		print('---end  : ',videoName)




