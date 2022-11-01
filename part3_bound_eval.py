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

def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(prefix + '[%s]-Step [%s/%s]-%s\r' % (bar, count, total, suffix))
    sys.stdout.flush()


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
        print('from', self.prefix)
        print(f'Number of images is {len(self.images)}')
    
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


def eval_non_ada_class_acc(model, data_loader, device):
    model.eval()
    class_corr_num = 0
    data_cnt = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            size = len(images)
            class_output = model(images)

            _, predicted_class = torch.max(class_output.data, 1)
            

            class_corr_num += (predicted_class == labels).sum().item()
            
            data_cnt += size

    print(class_corr_num, data_cnt)
    return class_corr_num / data_cnt

# def test(feature_extractor, class_classifier, domain_classifier, data_loader, device, input_type):
    
#     feature_extractor.eval()
#     class_classifier.eval()
#     domain_classifier.eval()

#     class_corr_num = 0
#     domain_corr_num = 0
#     data_cnt = 0
#     with torch.no_grad():
#         for _, (data, label,) in enumerate(tqdm(data_loader)):
#             data = data.to(device)
#             label = label.to(device)
#             if input_type == 'src':
#                 labels_domain = torch.zeros(len(data)).long().to(device)
#             elif input_type == 'tgt':
#                 labels_domain = torch.ones(len(data)).long().to(device)
#             # print('data:', data)
#             # dgdfsgf
#             feature = feature_extractor(data)
#             class_preds = class_classifier(feature)
#             # print(class_preds)
#             domain_preds = domain_classifier(feature, 0)
#             # print(domain_preds)
#             data_cnt += len(data)
            
#             # class_preds = class_preds.argmax(dim=1)
#             # domain_preds = domain_preds.argmax(dim=1)
#             # # correct if label == predict_label
#             # class_corr_num += (class_preds.eq(label.view_as(class_preds)).sum().item())
#             # domain_corr_num += (domain_preds.eq(labels_domain.view_as(domain_preds)).sum().item())
            
#             _, class_preds = torch.max(class_preds,1)
#             _, domain_preds = torch.max(domain_preds,1)
            
#             class_corr_num += (class_preds == label).sum().item()
#             domain_corr_num += (domain_preds == labels_domain).sum().item()
#     print(class_corr_num, domain_corr_num, data_cnt)
#     return class_corr_num / data_cnt, domain_corr_num / data_cnt

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--domain', default='', type=str)
    parser.add_argument('--model_file', type=str)

    args = parser.parse_args()
    fixed_seed(1)

    # os.makedirs(os.path.join('./part3_save_dir', args.domain, 'ckpt'), exist_ok=True)
    # save_path = os.path.join('./part3_save_dir', args.domain, 'ckpt')
    # log_path = os.path.join('./part3_save_dir', args.domain, 'acc_' + args.domain + '_.log')

    # init constants:
    # parser = create_parser()
    # args = parser.parse_args()

    # uid = str(uuid.uuid1())
    # best_epoch = 0
    # prev_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if (args.domain == 'mnist'):
        # mnist
        # train_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/val.csv', have_label=True, transform=tf)
    
    elif (args.domain == 'usps'):
        # usps
        # train_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/val.csv', have_label=True, transform=tf)
    
    elif (args.domain == 'svhn'):
        # svhn
        # train_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/val.csv', have_label=True, transform=tf)
    
    # src_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # max_step = max(len(src_train_dataloader), len(tgt_train_dataloader))
    # print(f'len(src_train_dataloader):{len(src_train_dataloader)}, len(tgt_train_dataloader):{len(tgt_train_dataloader)}')

    model = Non_Adaptive_model()
    load_parameters(model, args.model_file)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # eval model
    val_acc = eval_non_ada_class_acc(model, val_dataloader, device)
    # src_acc, src_acc_domain = eval_class_domain_acc(model, src_val_dataloader, device, "src")
    print('\nVal accuracy: {:.4f} '.format(val_acc))
    # print('SrcTestAcc: {:.4f}  SrcDomainAcc: {:.4f}'.format(src_acc, src_acc_domain))
    