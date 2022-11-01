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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--domain', default=.001, type=str)

    args = parser.parse_args()
    fixed_seed(1)

    os.makedirs(os.path.join('./part3_save_dir', args.domain, 'ckpt'), exist_ok=True)
    save_path = os.path.join('./part3_save_dir', args.domain, 'ckpt')
    log_path = os.path.join('./part3_save_dir', args.domain, 'acc_' + args.domain + '_.log')

    # init constants:
    # parser = create_parser()
    # args = parser.parse_args()

    # uid = str(uuid.uuid1())
    best_epoch = 0
    prev_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)


    # # step 1: prepare dataset
    # # MNIST-M → USPS / SVHN → MNIST-M / USPS → SVHN
    # mnist_dir = './hw2_data/digits/mnistm/data'
    # mnist_csv = './hw2_data/digits/mnistm/train.csv'
    # svhn_dir = './hw2_data/digits/svhn/data'
    # svhn_csv = './hw2_data/digits/svhn/train.csv'
    # usps_dir = './hw2_data/digits/usps/data'
    # usps_csv = './hw2_data/digits/usps/train.csv'

    # mnist_test_dir = './hw2_data/digits/mnistm/data'
    # mnist_test_csv = './hw2_data/digits/mnistm/val.csv'
    # svhn_test_dir = './hw2_data/digits/svhn/data'
    # svhn_test_csv = './hw2_data/digits/svhn/val.csv'
    # usps_test_dir = './hw2_data/digits/usps/data'
    # usps_test_csv = './hw2_data/digits/usps/val.csv'

    # src_dir = mnist_dir
    # src_csv = mnist_csv
    # test_src_dir = mnist_test_dir
    # test_src_csv = mnist_test_csv
    # tgt_dir = usps_dir
    # tgt_csv = usps_csv
    # test_dir = usps_test_dir
    # test_csv = usps_test_csv
    # if args.src_mode == "svhn":
    #     src_dir = svhn_dir
    #     src_csv = svhn_csv
    #     test_src_dir = svhn_test_dir
    #     test_src_csv = svhn_test_csv
    #     tgt_dir = mnist_dir
    #     tgt_csv = mnist_csv
    #     test_dir = mnist_test_dir
    #     test_csv = mnist_test_csv
    # elif args.src_mode == "usps":
    #     src_dir = usps_dir
    #     src_csv = usps_csv
    #     test_src_dir = usps_test_dir
    #     test_src_csv = usps_test_csv
    #     tgt_dir = svhn_dir
    #     tgt_csv = svhn_csv
    #     test_dir = svhn_test_dir
    #     test_csv = svhn_test_csv

    # src_dataset = DigitDataset(src_csv, src_dir)
    # if args.src_mode == "usps":
    #     src_dataset = DigitDataset(src_csv, src_dir, transform=transform)
    # src_train_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size,
    #                                              shuffle=False)

    # tgt_dataset = DigitDataset(tgt_csv, tgt_dir)
    # tgt_train_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batch_size,
    #                                              shuffle=False)

    # max_step = max(len(src_train_dataloader), len(tgt_train_dataloader))
    # print(f'src len:{len(src_train_dataloader)}, tgt len:{len(tgt_train_dataloader)}')

    # test_src_dataset = DigitDataset(test_src_csv, test_src_dir)
    # src_val_dataloader = torch.utils.data.DataLoader(test_src_dataset, batch_size=args.batch_size,
    #                                                   shuffle=False)


    if (args.domain == 'mnist'):
        # mnist
        train_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/mnistm/data', './hw2_data/digits/mnistm/val.csv', have_label=True, transform=tf)
    
    elif (args.domain == 'usps'):
        # usps
        train_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/usps/data', './hw2_data/digits/usps/val.csv', have_label=True, transform=tf)
    
    elif (args.domain == 'svhn'):
        # svhn
        train_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/train.csv', have_label=True, transform=tf)
        val_dataset = CustomImageDataset('./hw2_data/digits/svhn/data', './hw2_data/digits/svhn/val.csv', have_label=True, transform=tf)
    
    src_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    src_val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # max_step = max(len(src_train_dataloader), len(tgt_train_dataloader))
    # print(f'len(src_train_dataloader):{len(src_train_dataloader)}, len(tgt_train_dataloader):{len(tgt_train_dataloader)}')

    model = Non_Adaptive_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    start_epoch = 0
    # if args.resume:
    #     ckpt = load_checkpoint(args.ckpt)
    #     model.load_state_dict(ckpt['model'])
    #     start_epoch = ckpt['epoch'] + 1
    #     optimizer.load_state_dict(ckpt['optim'])
    #     uid = ckpt['uid']
    #     prev_acc = ckpt['tgtacc']

    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.cuda()

    #     print("Checkpoint restored, start from epoch {}.".format(start_epoch + 1))

    # step 5: main loop
    best_acc = 0.0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        # for i, src_data in enumerate(src_train_dataloader):
        for _, (data, label,) in enumerate(tqdm(src_train_dataloader)):
            
            # prepare data
            img_src, label_src = data.to(device), label.to(device)

            optimizer.zero_grad()

            # train on source domain
            src_class_output = model(img_src)
            class_loss = criterion(src_class_output, label_src)
            # src_loss = criterion(src_domain_output, domain_label_src)

            # # train on target domain
            # _, tgt_domain_output = model(img_tgt, alpha)
            # tgt_loss = criterion(tgt_domain_output, domain_label_tgt)

            ###
            # src_feature = feature_extractor(img_src)
            # class_preds = class_classifier(src_feature)
            # class_loss = criterion(class_preds, label_src)
            # src_preds = domain_classifier(src_feature, alpha)
            # src_loss = criterion(src_preds, domain_label_src)

            # tgt_feature = feature_extractor(img_tgt)
            # tgt_preds = domain_classifier(tgt_feature, alpha)
            # tgt_loss = criterion(tgt_preds, domain_label_tgt)
            

            # TODO: domain loss theta
            loss = class_loss
            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics
            # accum_loss += loss.item()
            # prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + args.epochs)
            # if (i + 1) % 10 == 0:  # print every 10 mini-batches
            #     suffix = 'Train Loss: {:.4f} SCL: {:.4f} SDL: {:.4f} TDL: {:.4f}'.format(
            #         accum_loss / (i + 1), class_loss, src_loss, tgt_loss)
            #     progress_bar(i + 1, max_step, prefix, suffix)

        # train_src_label_acc, train_src_domain_acc = test(feature_extractor, class_classifier, domain_classifier, src_val_dataloader, device, 'src')
        # train_tgt_label_acc, train_tgt_domain_acc = test(feature_extractor, class_classifier, domain_classifier, tgt_train_dataloader, device, 'tgt')
        # print('\nTrainAcc: {:.4f} TrainDomainAcc: {:.4f}'.format(train_src_label_acc, train_src_domain_acc))
        # print('SrcTestAcc: {:.4f}  SrcDomainAcc: {:.4f}'
        #       .format(train_tgt_label_acc, train_tgt_domain_acc))

        # eval model
        val_acc = eval_non_ada_class_acc(model, src_val_dataloader, device)
        # src_acc, src_acc_domain = eval_class_domain_acc(model, src_val_dataloader, device, "src")
        print('\nVal accuracy: {:.4f} '.format(val_acc))
        # print('SrcTestAcc: {:.4f}  SrcDomainAcc: {:.4f}'.format(src_acc, src_acc_domain))
        
        with open(log_path, 'a') as f :
            f.write(f'epoch = {epoch}\n')
            # f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            # f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val tgt class acc : {val_acc} \n' )
            if val_acc > best_acc:
                f.write(f'Find best, save best model at epoch {epoch}!\n')
            f.write('============================\n')

        # save model for every epoch 
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pt'))
        if val_acc > best_acc:
            print('Save best model at epoch', epoch)
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            early_stop_cnt = 0


        # if early_stop_cnt > early_stop:
        #     print('early stop!')
        #     break
        # # step 6: save checkpoint if better than previous
        # if src_acc > prev_acc:
        #     checkpoint = {
        #         'model': model.state_dict(),
        #         'epoch': epoch,
        #         'optim': optimizer.state_dict(),
        #         'uid': uid,
        #         # 'tgtacc': tgt_acc,
        #         'srcacc': src_acc,
        #     }
    #         save_checkpoint(checkpoint,
    #                         os.path.join(args.ckpt_path, f"{args.src_mode}-{uid[:8]}.pt"))
    #         print(f'Epoch {epoch + 1} Saved!')
    #         prev_acc = src_acc
    #         best_epoch = epoch + 1

    #         step_count += 1

    # # step 7: logging experiment
    # experiment_record_p2("./ckpt/p3/p3_log.txt",
    #                      uid,
    #                      time.ctime(),
    #                      args.batch_size,
    #                      args.lr,
    #                      best_epoch,
    #                      prev_acc)
