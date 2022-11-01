import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from part3_model import *
from PIL import Image
from torch.utils.data.dataset import Dataset
import argparse
# import sys
import glob

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")


def fixed_seed(myseed):
    np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

class CustomImageDataset_with_filename(Dataset):
    def __init__(self, data_folder_path, transform=None):
        if(data_folder_path[-1] != '/'):
            data_folder_path += '/'
        images_filename = glob.glob(data_folder_path+'*.png')
        images_filename.sort()
        
            
        self.images = images_filename
        self.transform = transform
        self.prefix = data_folder_path
        print('from', self.prefix)
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        full_path = os.path.join(self.images[idx])
        img = Image.open(full_path).convert("RGB")
        transform_img = self.transform(img)
        # if self.labels != None:
        #     return (transform_img, self.labels[idx], self.images[idx])
        # else:
        return (transform_img, self.images[idx])

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--usps_model_file', default='', type=str)
    parser.add_argument('--svhn_model_file', default='', type=str)

    
    args = parser.parse_args()

    
    # fixed random seed
    fixed_seed(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    """ training hyperparameter """

    model = myDANN()
    
    # Put model's parameters on your device
    
    # print(model)
    if args.input_dir.find('usps')!= -1:
        load_parameters(model, args.usps_model_file)
    elif args.input_dir.find('svhn')!= -1:
        load_parameters(model, args.svhn_model_file)

    model = model.to(device)
    
    val_set = CustomImageDataset_with_filename(args.input_dir, transform=tf)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        corr_num = 0
        val_acc = 0.0
        
        for batch_idx, (data, fname,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            # label = label.to(device)

            # pass forward function define in the model and get output 
            output, _ = model(data, 0) 

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            results.append((fname[0].split('/')[-1], str(int(pred[0]))))
            
    with open(args.output_file, 'w') as f:
        f.write('image_name,label\n')
        f0, p0 = results[0]
        f.write(f0)
        f.write(',')
        f.write(p0)
        for fname, predl in results[1:]:
            f.write('\n')
            f.write(fname)
            f.write(',')
            f.write(predl)
        