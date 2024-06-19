import audio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from params_data import *

def _preprocess_speaker_dirs(dataset_dir,output_dir):
    min_frame = 324123
    max_frame = 0
    source_file = dataset_dir+"/"+"_source.txt"
    fptr = open(source_file,'r')
    lis = fptr.readlines()
    for i in lis:
        temp = i.split(',')
        np_file_name = temp[0]
        audio_file = temp[1]
        audio_file = audio_file.rstrip('\n')
        wav = audio.preprocess_wav(audio_file)
        if len(wav) == 0:
            print(audio_file)
            continue
        
        # Create the mel spectrogram, discard those that are too short
        # frames = audio.wav_to_mel_spectrogram(wav)
        frames = audio.wav_to_spectrogram(wav)
        # if len(frames) < partials_n_frames: # 300 frames
        #     continue
        # print(type(frames))
        print(frames.shape)
        min_frame = min(min_frame,frames.shape[0])
        max_frame = max(max_frame,frames.shape[0])
        out_file_path  = output_dir+"/"+np_file_name
        np.save(out_file_path, frames)
        print(np_file_name,"completed")
    fptr.close()
    print("Min frame:", min_frame," Max frame:",max_frame)
    

dataset_dir = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/SWAN-DF/test/all_real_videos_256x256_30_audio"

output_dir = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/SWAN-DF/test/all_real_videos_256x256_30_audio_numpy"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

_preprocess_speaker_dirs(dataset_dir,output_dir)