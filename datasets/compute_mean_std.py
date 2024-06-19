#from data_objects.speaker import Speaker
from pathlib import Path
from pyexpat import features
from speaker import Speaker
import numpy as np
import os

def compute_mean_std(dataset_dir, output_path_mean, output_path_std):
    print("Computing mean std...")
    # speaker_dirs = [f for f in dataset_dir.glob("*") if f.is_dir()]

    # if len(speaker_dirs) == 0:
    #     raise Exception("No speakers found. Make sure you are pointing to the directory "
    #                     "containing all preprocessed speaker directories.")
    # #speaker_dirs = dataset_dir
    # speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
    # sources = []
    # for speaker in speakers:
    #     sources.extend(speaker.sources)
    sources =  os.listdir(dataset_dir)
    sumx = np.zeros(257, dtype=np.float32)
    sumx2 = np.zeros(257, dtype=np.float32)
    count = 0
    n = len(sources)
    for i, source in enumerate(sources):
        source_path = os.path.join(dataset_dir,source)
        #feature = np.load(source[0].joinpath(source[1]))
        feature = np.load(source_path)
        sumx += feature.sum(axis=0)
        sumx2 += (feature * feature).sum(axis=0)
        count += feature.shape[0]

    mean = sumx / count
    std = np.sqrt(sumx2 / count - mean * mean)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    output_path_mean = output_path_mean
    output_path_mean = output_path_mean
    np.save(output_path_mean, mean)
    np.save(output_path_std, std)
    print(output_path_mean)
    print(output_path_std)
    print("computed mean and std for this iteration")
    

dataset_dir = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/fakeavceleb_partition_2/test_run/nturgb+d_rgb_256x256_30_audio"
print(dataset_dir)
output_path_mean = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/fakeavceleb_partition_2/test_run/mean.npy"
output_path_std = "/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data/codes/vinod/fakeavceleb_partition_2/test_run/std.npy"
compute_mean_std(dataset_dir, output_path_mean, output_path_std)