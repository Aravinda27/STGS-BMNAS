from __future__ import print_function
from pathlib import Path
from typing import Any

import numpy as np

from torchvision import transforms as T
from datasets.transforms import Normalize, TimeReverse
#from transforms import Normalize, TimeReverse

class audio_feature():
    def __init__(self, root,data_path,partial_n_frames):
        #print("Total no of partial n frames:",partial_n_frames)
        self.root = root
        self.data_path = data_path
        self.partial_n_frames = partial_n_frames
        mean = np.load(self.root+"/"+'mean.npy')
        std = np.load(self.root+"/"+'std.npy')
        self.feature = np.load(data_path)
        self.transform = T.Compose([
            Normalize(mean, std),
            TimeReverse(),
        ])
        
    def __call__(self):
        if self.feature.shape[0] <= self.partial_n_frames:
            start = 0
            while self.feature.shape[0] < self.partial_n_frames:
                self.feature = np.repeat(self.feature, 2, axis=0)
        else:
            start = np.random.randint(0, self.feature.shape[0] - self.partial_n_frames)
        end = start + self.partial_n_frames
        out_feature =  self.feature[start:end]
        out_feature = self.transform(out_feature)
        
        return out_feature

# root = "/home/mt0/22CS60R39/BM-NAS_dataset/NTU/"
# data_path= "/home/mt0/22CS60R39/BM-NAS_dataset/NTU/nturgb+d_rgb_256x256_30_audio/00109_id00391_wavtolip.npy"
# partial_n_frames = 1019 
# x = DeepSpeakerDataset(Path(root),data_path, x``)
# print(x.shape)