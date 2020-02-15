import os
from config import  config
from tqdm import tqdm
import random

if not os.path.exists(r'..\datalist'):
    os.makedirs(r'..\datalist')
    
train_txt = open(r'..\datalist\train.txt', 'w')
val_txt = open(r'..\datalist\valid.txt', 'w')
label_txt = open(r'..\datalist\label_list.txt', 'w')

label_list = []

for dir in tqdm(os.listdir(config.data_root)):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list) - 1)))
        data_path = os.path.join(config.data_root, dir)
        train_list = random.sample(os.listdir(data_path), int(len(os.listdir(data_path)) * 0.8))

        for im in train_list:
            train_txt.write('{}\\{} {}\n'.format(dir, im, str(len(label_list) - 1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list) - 1)))