import os 
import pdb 
import csv 
import numpy as np 
import torch
from tqdm import tqdm
import pdb
from dv import AedatFile



data_path = r"/''''/Poker_Event/"
save_path = r"/''''/Poker_Event_to_tensor/"
video_files = os.listdir(data_path)

n_clips=16
t_gap_set = 12

if __name__ == '__main__':
	device = torch.device("cuda:0")
	for fileID in range(len(video_files)):
		videoName = video_files[fileID]

		fileLIST = os.listdir(data_path + videoName + r'/')
		save_tensor_Event = save_path + videoName + r'/'


		if not os.path.exists(save_tensor_Event):
			os.makedirs(save_tensor_Event)
		Ratio_of_coincidence= []
		for newFileID in tqdm(range(len(fileLIST))):
			csv_videoName = fileLIST[newFileID]
			# print('---start  : ',csv_videoName)
			event_tensor_Event = save_tensor_Event +csv_videoName[:-7]+'/'


			if os.path.exists(event_tensor_Event):
				continue
			if not os.path.exists(event_tensor_Event):
				os.makedirs(event_tensor_Event)

			aedat_file_path = data_path + videoName + r'/' + csv_videoName
			with AedatFile(aedat_file_path) as f:
				# print(f.names)
				events = np.hstack([packet for packet in f['events'].numpy()])
				# print(events)
				t, x, y, p = events['timestamp'], events['x'], events['y'], events['polarity']
				time_length = t[-1]-t[0]
				t_gap11 = time_length/n_clips
				t_gap = int((events['timestamp'][-1]-events['timestamp'][0])/n_clips) 
				index_t = torch.tensor(t)
				for i in range(n_clips):
					idx_1 = torch.where((events['timestamp'][0]+i*t_gap)<index_t)
					idx_2 = torch.where(index_t<(events['timestamp'][0]+(i+1)*t_gap))	
					idx_start= idx_1[0][0]
					idx_end= idx_2[0][-1]	
					x_clip = x[idx_start:idx_end]
					y_clip = y[idx_start:idx_end]
					t_clip0 = t[idx_start:idx_end]	
					t_clip = ((t_clip0-t_clip0[0]) / (t_clip0[-1]-t_clip0[0])) * (t_gap_set-1.0)
					# x = x.astype(np.float)
					# y = y.astype(np.float)

					events_index = torch.LongTensor(np.vstack((x_clip,y_clip,t_clip)))
					events_index = tuple(events_index)
					tensor_Event = torch.zeros(346,260,t_gap_set) 
					tensor_Event_save=tensor_Event.index_put(events_index,  values = torch.tensor(1.))

					# coincidence=((len(x_clip)-tensor_Event_save.sum())/len(x_clip)).tolist()
					# Ratio_of_coincidence.append(coincidence)
					# if coincidence>0.4:
					# 	print('singal_coincidence： {:.2%}'.format(coincidence))

					np.savez_compressed(event_tensor_Event+'{:04d}'.format(i)+'.npz',tensor_Event = tensor_Event_save)

		# print('Ratio_of_coincidence——mean： {:.2%}'.format(np.array(Ratio_of_coincidence).mean()))
		# print('Ratio_of_coincidence——max： {:.2%}'.format(np.array(Ratio_of_coincidence).max()))
			# print('---end  : ',csv_videoName)
		print('---end  : ',videoName)




