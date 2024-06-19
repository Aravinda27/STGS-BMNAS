
import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
from sklearn.utils import shuffle
import torchvision.transforms as transforms
from IPython import embed
#from datasets.DeepSpeaker2 import audio_feature
from datasets.DeepSpeaker2 import audio_feature
import numpy as np
import random


def load_video(path, vid_len=24):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Init the numpy array
    video = np.zeros((vid_len, height, width, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)
    np_idx = 0
    cnt = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        if cap.isOpened() and fr_idx in taken:
            video[np_idx, :, :, :] = frame.astype(np.float32)
            np_idx += 1
        if np_idx == vid_len:
            break
    cap.release()
    return video

def load_video_npy(path):
    video = np.load(path)
    video = video.astype(np.float32)
    return video

# 3d coordinates cf. https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/read_skeleton_file.m for more details

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        rgb, spec ,label = sample['rgb'], sample['spec'] ,sample['label']
        spec = sample['spec']
        tensor_spec = torch.from_numpy(spec.astype(np.float32))
        tensor_spec = tensor_spec.view(8,256,257)
        tensor_spec = tensor_spec.unsqueeze(3)
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'spec': tensor_spec,
                'label': torch.from_numpy(np.asarray(label))}
    
class NormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb, spec ,label = sample['rgb'], sample['spec'] ,sample['label']
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            #print("IN normalize frame",num_frames_rgb)
            indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
            rgb = rgb[indices_rgb, :, :, :]
        
        return {'rgb': rgb,
                'spec':spec,
                'label': label}
    

class AugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, spec, label = sample['rgb'], sample['spec'], sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
            rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb), :, :, :]

        # if spec.shape[0] != 1:
        #     valid_size = spec.shape[1]
        #     p = np.random.rand(1) * (1.0 - self.p_interval) + self.p_interval
        #     cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
        #     bias = np.random.randint(0, valid_size - cropped_length + 1)
        #     spec = spec[:, bias:bias + cropped_length, :, :]

        return {'rgb': rgb,
                'spec': spec,
                'label': label}


class NTU(Dataset):
    def __init__(self, root_dir='',  # /home/juanma/Documents/Data/ROSE_Action
                 transform=None,
                 stage='train',
                 args=None,
                 vid_len=(8, 32),
                 vid_dim=256,
                 vid_fr=30):

        # basename_rgb = os.path.join(root_dir, 'nturgb+d_rgb_{0}x{0}_{1}'.format(vid_dim, vid_fr))
        # load .npy directly
        if(stage == "train_val"):
            stage_list = ['train','dev']
            temp_root_dir = root_dir
            rgb_list = []
            spec_list = []
            label_dict = {}
            print("################ Inside train_val ####################")
            for i in range(len(stage_list)):
                print("In dataset/ ntu in bm nas model")
                print("Root_dir path:",root_dir)
                root_dir = temp_root_dir
                root_dir = os.path.join(root_dir, stage_list[i])
                basename_rgb = os.path.join(root_dir, 'nturgb+d_rgb_{0}x{0}_{1}'.format(vid_dim, vid_fr))
                basename_spec = os.path.join(root_dir, 'nturgb+d_rgb_256x256_30_audio')
                basename_label = os.path.join(root_dir, 'label2.txt')
                print("RGB video Path: ",basename_rgb)
                print("spectogram path:",basename_spec)
                self.original_w, self.original_h = 1920, 1080
                self.vid_len = vid_len
                rgb_list +=[os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                            f.split(".")[-1] == "mp4"]
                spec_list += [os.path.join(basename_spec, f) for f in sorted(os.listdir(basename_spec)) if
                            f.split(".")[-1] == "npy"]
                label_file = open(basename_label,'r')
            
                for i in label_file.readlines():
                    temp = i.rstrip("\n")
                    temp = temp.split(" ")
                    label_dict[temp[0]] = int(temp[1])
                print("*******Dict len: ", len(label_dict))
            print("RBG_list len: ",len(rgb_list))
            print("Spec list len: ", len(spec_list))
            print("Dict len: ", len(label_dict))
            self.rgb_list = []
            self.spec_list = []
            self.labels = []
            # self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
            #                 f.split(".")[-1] == "mp4"]
            # self.spec_list += [os.path.join(basename_spec, f) for f in sorted(os.listdir(basename_spec)) if
            #                 f.split(".")[-1] == "npy"]
            self.rgb_list += rgb_list
            self.spec_list+= spec_list
            for i in self.rgb_list:
                temp = i.split("/")
                #print(temp)
                #print(temp[-1]," ", label_dict[temp[-1]])
                self.labels.append(label_dict[temp[-1]])
            print("Len of self labels:",len(self.labels))
        
        else:
            print("In dataset/ ntu in bm nas model")
            print("Root_dir path:",root_dir)
            root_dir = os.path.join(root_dir, stage)
            basename_rgb = os.path.join(root_dir, 'nturgb+d_rgb_{0}x{0}_{1}'.format(vid_dim, vid_fr))
            basename_spec = os.path.join(root_dir, 'nturgb+d_rgb_256x256_30_audio')
            basename_label = os.path.join(root_dir, 'label2.txt')
            print("RGB video Path: ",basename_rgb)
            print("spectogram path:",basename_spec)
            self.original_w, self.original_h = 1920, 1080
            self.vid_len = vid_len

            self.rgb_list = []
            self.spec_list = []
            self.labels = []

            self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                            f.split(".")[-1] == "mp4"]
            self.spec_list += [os.path.join(basename_spec, f) for f in sorted(os.listdir(basename_spec)) if
                            f.split(".")[-1] == "npy"]
            label_file = open(basename_label,'r')
            label_dict = {}
            for i in label_file.readlines():
                temp = i.rstrip("\n")
                temp = temp.split(" ")
                label_dict[temp[0]] = int(temp[1])
            for i in self.rgb_list:
                temp = i.split("/")
                #print(temp)
                #print(temp[-1]," ", label_dict[temp[-1]])
                self.labels.append(label_dict[temp[-1]])
        
        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.mode = stage

        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        #skepath = self.ske_list[idx]
        
        spectogrampath = self.spec_list[idx]

        label = self.labels[idx]
        #print("Label type",type(label))
        #print(label)
        video = np.zeros([1])
        #skeleton = np.zeros([1])

        if self.args.modality == "rgb" or self.args.modality == "both":
            # video = load_video(rgbpath)
            video = load_video(rgbpath)
        audio_object = audio_feature(self.root_dir,spectogrampath,2048)
        spectogram = audio_object()
        #print("spectrogram shape :",spectogram.shape)
        video = self.video_transform(self.args, video)
        sample = {'rgb': video, 'spec':spectogram, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def video_transform(self, args, np_clip):
        if args.modality == "rgb" or args.modality == "both":
            # Div by 255
            np_clip /= 255.
            # Normalization
            np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
            np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std
        return np_clip


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="store",
                        dest="folder",
                        help="Path to the data",
                        default="/home/mt0/22CS60R39/BM-NAS_dataset/NTU/")
    parser.add_argument('--outputdir', type=str, help='output base dir', default='/home/mt0/22CS60R39/video_checkpoint/')
    parser.add_argument('--datadir', type=str, help='data directory', default='NTU')
    parser.add_argument("--j", action="store", default=12, dest="num_workers", type=int,
                        help="Num of workers for dataset preprocessing ")

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, help="length of video")
    parser.add_argument('--modality', type=str, help='modality: rgb, skeleton, both', default='both')
    parser.add_argument("--hp", action="store_true", default=False, dest="hp", help="random search on hp")
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument('--num_classes', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument("--clip", action="store", default=None, dest="clip", type=float,
                        help="if using gradient clipping")
    parser.add_argument("--lr", action="store", default=0.001, dest="learning_rate", type=float,
                        help="initial learning rate")
    parser.add_argument("--lr_decay", action="store_true", default=False, dest="lr_decay",
                        help="learning rate exponential decay")
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")
    parser.add_argument('--epochs', type=int, help='training epochs', default=10)

    args = parser.parse_args()
    

    train_transformer = transforms.Compose([NormalizeLen(), ToTensor()])
    dataset = NTU(args.folder, train_transformer, 'train', args=args)
    print("Here after data preparation")
    iterator = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True,
                                           num_workers=args.num_workers)
    for batch in iterator:
        print("RGB", batch['rgb'].shape,"SPEC",batch['spec'].shape,", label", batch['label'].shape)


"""
python ntu.py --folder=/mnt/data/xiaoxiang/yihang/datasets/

    --parallel
    --num_workers

"""
