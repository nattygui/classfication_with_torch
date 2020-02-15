import os
from PIL import Image

from torch.utils.data.dataset import Dataset

from config import config


def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            ims.append(im)
            labels.append(int(label))
    return ims, labels


class RSDataset(Dataset):
    def __init__(self, txt_path, transform):
        self.ims, self.labels = read_txt(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        im_path = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(config.data_root, im_path)
        im = Image.open(im_path)
        if self.transform:
            im = self.transform(im)

        return im, label

    def __len__(self):
        return len(self.ims)
