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
import sys
import glob

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")


class CustomImageDataset(Dataset):
    def __init__(self, data_folder_path, file_label_csv, have_label=True, transform=None):
        img_names = []
        img_labels = []
        with open(file_label_csv, 'r') as f:
            imgs_and_labels = [line.rstrip().split(',') for line in f]
            img_names = [i[0] for i in imgs_and_labels[1:]]
            if have_label:
                img_labels = [int(i[1]) for i in imgs_and_labels[1:]]
                self.labels = torch.tensor(img_labels)
            else:
                self.labels = None
            
        self.images = img_names
        self.transform = transform
        self.prefix = data_folder_path
        # print('from', self.prefix)
        # print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        full_path = os.path.join(self.prefix, self.images[idx])
        img = Image.open(full_path).convert("RGB")
        transform_img = self.transform(img)
        if self.labels != None:
            return (transform_img, self.labels[idx])
        else:
            return (transform_img)


def fixed_seed(myseed):
    np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def eval_class_domain_acc(model, data_loader, device, input_type):
    model.eval()
    class_corr_num = 0
    domain_corr_num = 0
    data_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            size = len(images)
            if input_type == 'tgt':
                labels_domain = torch.ones(size).long().to(device)
            elif input_type == 'src':
                labels_domain = torch.zeros(size).long().to(device)
            class_output, domain_output = model(images, alpha=0)

            _, predicted_class = torch.max(class_output.data, 1)
            _, predicted_domain = torch.max(domain_output.data, 1)

            class_corr_num += (predicted_class == labels).sum().item()
            domain_corr_num += (predicted_domain == labels_domain).sum().item()

            data_cnt += size
    print(class_corr_num, domain_corr_num, data_cnt)
    return class_corr_num / data_cnt, domain_corr_num / data_cnt

  

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
    
    parser.add_argument('--model_type', default='', type=str)
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
    if args.model_type.find('usps')!= -1:
        load_parameters(model, args.usps_model_file)
    elif args.model_type.find('svhn')!= -1:
        load_parameters(model, args.svhn_model_file)

    model = model.to(device)
    
    val_set = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/val.csv', have_label=True, transform=tf)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    last_second_results = []
    labels = []
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        corr_num = 0
        val_acc = 0.0
        data_len = 0
        ## TO DO ## 
        # Finish forward part in validation. You can refer to the training part 
        # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

        for batch_idx, (data, label,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)
            data_len += len(data)
            
            output, _ = model(data, 0)
            pred = output.argmax(dim=1)
            
            corr_num += (pred.eq(label.view_as(pred)).sum().item())
        print(corr_num/data_len)