from torch.utils.data import Dataset
import torchaudio
import torch
import numpy as np
import os

class My_Dataset(Dataset):
    def __len__(self):
        return 7350
    
    def __getitem__(self, index):
        audio, sample_rate = torchaudio.load(os.path.join('samples', 'Merge', 'Merge-' + str(index + 1) + '.wav'))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = transform(audio)[:, : 4*16000]
        ans = np.loadtxt("./ans.txt")[index]
        return audio, ans