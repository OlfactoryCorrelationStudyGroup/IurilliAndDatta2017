#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import scipy
from scipy import io
from scipy.sparse import csc_array


#class IurilliAndDatta2017(Dataset):

    #def __init__(self, root_dir, transform=None):
        
#         self.root_dir = root_dir
#         self.transform = transform

#         # Assume images are in subfolders named by class (e.g., '0/', '1/')
#         self.image_paths = []
#         self.labels = []

#         # Map folder name to integer label
#         self.class_to_idx = {
#             cls_name: i for i, cls_name in enumerate(sorted(os.listdir(root_dir)))
#             if os.path.isdir(os.path.join(root_dir, cls_name))
#         }

#         for cls_name, cls_idx in self.class_to_idx.items():
#             cls_dir = os.path.join(root_dir, cls_name)
#             for f in os.listdir(cls_dir):
#                 if f.endswith(('.png', '.jpg', '.jpeg')):
#                     self.image_paths.append(os.path.join(cls_dir, f))
#                     self.labels.append(cls_idx)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]

#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # DATA_DIR = './data'
# # os.makedirs(os.path.join(DATA_DIR, '0'), exist_ok=True)
# # os.makedirs(os.path.join(DATA_DIR, '1'), exist_ok=True)
# # (e.g., Place images like data/0/cat_1.jpg, data/1/dog_1.jpg)
# # ---------------------------------------------------

# # --- 3. DATALOADER SETUP ---
# DATA_DIR = './data' # Assuming you created the 'data' directory
# BATCH_SIZE = 4
# N_CLASSES = 2 # Assuming two classes: '0' and '1'

# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Create Dataset and DataLoader
# try:
#     dataset = ImageFolderDataset(root_dir=DATA_DIR, transform=transform)
#     data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# except FileNotFoundError:
#     print("!!! ERROR: Please create the './data' directory with subfolders '0' and '1' containing images first.")
#     exit()



def get_data(file_path):

    mat_data = io.loadmat(file_path,squeeze_me=True,struct_as_record=False)

    # Access a specific variable
    return mat_data['espe']


def get_firing_rate_2(data,baseline_start=2000,resp_start=4000,resp_end=10000):

    baseline_duration = (resp_start - baseline_start)/1000 # in seconds
    resp_duration = (resp_end - resp_start)/1000
    spikes_dict = data.todok()
    rates = []
    for trial in range(10):
    
        if trial not in set(data.indices):
            rates.append(0.0)
            continue

        dok_arr = spikes_dict[trial].todok()
        keys = np.array(list(dok_arr.keys()))
        base_keys = keys[(keys>=baseline_start) & (keys<resp_start)]
        resp_keys = keys[(keys>=resp_start) & (keys<=resp_end)]
        base_spikes = []
        resp_spikes = []
        
        for key in base_keys:
            base_spikes.append(dok_arr[key])
        for key in resp_keys:
            resp_spikes.append(dok_arr[key])
        
        base_rate = np.array(base_spikes).sum()/baseline_duration
        resp_rate = np.array(resp_spikes).sum()/resp_duration

        if base_rate > resp_rate:
            rates.append(0.0)
        else:
            rates.append(resp_rate-base_rate)


    return np.array(rates)


def preprocess_2(data,baseline_start=2000,resp_start=4000,resp_end=10000):

    sessions = []
    for session in data:
        cols = np.empty(shape=(150,0)) # n odors * n trials = 150 / need to find a better way to initialize
        for shank in session.shank:
            if type(shank.SUA)==np.ndarray: # if SUA is empty it appears as an empty np array
                continue
            if type(shank.SUA.cell)!=np.ndarray: # if there is only one cell it appears as mat_struct
                cells = np.array([shank.SUA.cell])
            else:
                cells = shank.SUA.cell
                for cell in cells:
                    rows = np.array([]) # rows per 1 neuron, should result in (150,) np array
                    for odor in cell.odor:
                        spikes = odor.spikeMatrix # scipy.sparse._csc.csc_array
                        rates = get_firing_rate_2(spikes,baseline_start,resp_start,resp_end) # (10,) np array
                        rows = np.concatenate((rows,rates),axis=0)
                    
                    rows = rows[:,np.newaxis] # converts rows to (150,1) shape
                    cols = np.concatenate((cols,rows),axis=1)

        sessions.append(cols)     
        #print('Session appended')  

    return sessions




file_path = r'D:\!Studying\NeuroData\ML\final_project_ML\group_github\IurilliAndDatta2017\data\aPCx_15.mat'

data = get_data(file_path=file_path)

#print(data[0].shank[0].SUA.cell[0].odor[0].spikeMatrix)
#print(data[0].shank[0].SUA.cell[1]._fieldnames)
#print(len(data))

# setting resp_end=6000 should give the same results as Ofek's non-normalized .mat data
# unless i fucked up somewhere
sessions = preprocess_2(data) 

for session in sessions:

    print(f'Shape: {session.shape}')