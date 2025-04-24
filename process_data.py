import os
import torch
import pickle
import numpy as np
import torch.nn.functional as F

from moviepy import *
from pathlib import Path

def load_video(video_path):
    clip = VideoFileClip(video_path)
    # print(clip.size)
    print(video_path, clip.fps)
    frame_list = []
    for frame_number, frame in enumerate(clip.iter_frames()):
        frame_array = np.array(frame)
        frame_list.append(frame_array)
    clip.close()

    video_data = np.stack(frame_list)

    print(video_data.shape)
    return video_data

def load_normal_user_data(user_dir, sample_frame_num):
    user_data = []
    for video_name in ["A1.avi", "B1.avi", "C1.avi", "D1.avi"]:
        video_path = os.path.join(user_dir, video_name)
        if os.path.exists(video_path):
            video_data = load_video(video_path)
        else:
            print("%s empty" % video_path)
            video_data = np.zeros((sample_frame_num, 100, 100, 3))

        video_data = torch.tensor(video_data).float()
        video_data = video_data.permute(3, 0, 1, 2).unsqueeze(dim=0)
        # 708 508
        video_data = F.interpolate(video_data, (40, 256, 256), mode="trilinear")[0].permute(1, 2, 3, 0)
        video_data = video_data.numpy().astype(np.float32)

        user_data.append(video_data)
    user_data = np.stack(user_data)
    print(user_data.shape)
            
    return user_data

def load_abnormal_user_data(user_dir, sample_frame_num):
    user_data = []
    for video_name in ["A1.avi", "B1.avi", "C1.avi", "D1.avi"]:
        video_path = os.path.join(user_dir, video_name)
        if os.path.exists(video_path):
            video_data = load_video(video_path)
        else:
            print("%s empty" % video_path)
            video_data = np.zeros((sample_frame_num, 872, 1896, 3))

        video_data = torch.tensor(video_data).float()
        video_data = video_data.permute(3, 0, 1, 2).unsqueeze(dim=0)
        # 872 1896
        video_data = F.interpolate(video_data, (40, 256, 256), mode="trilinear")[0].permute(1, 2, 3, 0)
        video_data = video_data.numpy().astype(np.float32)

        user_data.append(video_data)
    user_data = np.stack(user_data)
            
    return user_data

data_root = Path("/home/wangnannan/workdir/as/all")
save_dir = Path("/home/wangnannan/workdir/as/pkl")
save_dir.mkdir(parents=True,exist_ok=True)

for folder in data_root.iterdir():
    for case_folder in folder.iterdir():
        status = folder.name.split('_')[0]
        
        save_path = save_dir / (case_folder.name + f"_{status}.pkl")
        if status == 'normal':
            user_data = load_normal_user_data(case_folder, sample_frame_num=2)
        else:
            user_data = load_abnormal_user_data(case_folder, sample_frame_num=2)
            
        with open(save_path, "wb") as f:
            pickle.dump(user_data, f)
