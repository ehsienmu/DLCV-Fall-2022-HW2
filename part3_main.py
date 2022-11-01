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
        for _, (images, labels,) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.to(device)
            size = len(images)

            if input_type == 'tgt':
                labels_domain = torch.ones(size).long().to(device)
            elif input_type == 'src':
                labels_domain = torch.zeros(size).long().to(device)
            class_output, domain_output = model(images, 0)

            pred_class = torch.argmax(class_output.data, dim=1)
            pred_domain = torch.argmax(domain_output.data, dim=1)

            class_corr_num += (pred_class.eq(labels.view_as(pred_class)).sum().item())
            domain_corr_num += (pred_domain.eq(labels_domain.view_as(pred_domain)).sum().item())

            data_cnt += size
            
    return class_corr_num / data_cnt, domain_corr_num / data_cnt

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--target_domain', default='svhn', type=str)

    args = parser.parse_args()
    fixed_seed(1)
    folder_name = 'mnistm_' + args.target_domain
    os.makedirs(os.path.join('./part3_save_dir', folder_name, 'ckpt'), exist_ok=True)
    save_path = os.path.join('./part3_save_dir', folder_name, 'ckpt')
    log_path = os.path.join('./part3_save_dir', folder_name, 'acc_' + folder_name + '_.log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # mnist
    src_train_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/train.csv', have_label=True, transform=tf)
    src_val_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/val.csv', have_label=True, transform=tf)
    
    if (args.target_domain == 'usps'):
        # usps
        tgt_train_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/train.csv', have_label=True, transform=tf)
        tgt_val_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/val.csv', have_label=True, transform=tf)
    
    elif (args.target_domain == 'svhn'):
        # svhn
        tgt_train_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/train.csv', have_label=True, transform=tf)
        tgt_val_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/val.csv', have_label=True, transform=tf)
    
    src_train_dataloader = DataLoader(src_train_dataset, batch_size=args.batch_size, shuffle=True)
    tgt_train_dataloader = DataLoader(tgt_train_dataset, batch_size=args.batch_size, shuffle=True)

    src_val_dataloader = DataLoader(src_val_dataset, batch_size=args.batch_size, shuffle=False)
    tgt_val_dataloader = DataLoader(tgt_val_dataset, batch_size=args.batch_size, shuffle=False)

    max_step = max(len(src_train_dataloader), len(tgt_train_dataloader))

    model = myDANN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        
        src_iter = iter(src_train_dataloader)
        for i, tgt_data in enumerate(tgt_train_dataloader):
            try:
                src_data = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_dataloader)
                src_data = next(src_iter)

            p = float(i + epoch * max_step) / args.epochs / max_step
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            img_src, label_src = src_data[0].to(device), src_data[1].to(device)
            img_tgt = tgt_data[0].to(device)
            
            size_src = len(img_src)
            size_tgt = len(img_tgt)
            domain_label_src = torch.zeros(size_src).long().to(device)
            domain_label_tgt = torch.ones(size_tgt).long().to(device)

            optimizer.zero_grad()

            src_class_output, src_domain_output = model(img_src, alpha)
            _, tgt_domain_output = model(img_tgt, alpha)
            class_loss = criterion(src_class_output, label_src)
            src_loss = criterion(src_domain_output, domain_label_src)
            tgt_loss = criterion(tgt_domain_output, domain_label_tgt)

            loss = class_loss + src_loss + tgt_loss
            
            loss.backward()
            optimizer.step()

        tgt_val_acc, tgt_val_acc_domain = eval_class_domain_acc(model, tgt_val_dataloader, device, "tgt")
        # src_acc, src_acc_domain = eval_class_domain_acc(model, src_val_dataloader, device, "src")
        print('\nEpoch {}: Target_class_accuracy: {:.4f} Target_domain_accuracy: {:.4f}'.format(epoch, tgt_val_acc, tgt_val_acc_domain))
        # print('SrcTestAcc: {:.4f}  SrcDomainAcc: {:.4f}'.format(src_acc, src_acc_domain))
        
        with open(log_path, 'a') as f :
            f.write(f'epoch = {epoch}\n')
            # f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            # f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val tgt class acc : {tgt_val_acc}  val tgt domain = {tgt_val_acc_domain}\n' )
            if tgt_val_acc > best_acc:
                f.write(f'find best, save best model at epoch {epoch}!\n')
            f.write('============================\n')

        # save model for every epoch 
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pt'))
        if tgt_val_acc > best_acc:
            print('Save best model at epoch', epoch)
            best_acc = tgt_val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            early_stop_cnt = 0
