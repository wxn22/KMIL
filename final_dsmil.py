from dsmil_model import FCLayer, BClassifier, MILNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import glob
import os
import h5py
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, classification_report



def parse():
    parser = argparse.ArgumentParser(description='Train for dsmil')
    parser.add_argument('--epochs', type=int, default=100)

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=1024, help="The dimension of instance-level representations")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Dataset parameters
    parser.add_argument('--src_dir', type=str, default='/data_sdd/wxn/wocleansex_clean_var_0.9_ex_3', help='data path')
    parser.add_argument('--split_dir', type=str, default='/data_sdc/zrl/code/dtfd/split_balance', help='split path')
    parser.add_argument('--device', type=str, default='cuda:2', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--continue_fold', type=int, default=0)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--output_dir', default='/data_sdb/wxn/dsmil/result_var_0.9_2cls+_ex',
                        help='path where to save, empty for no saving')

    return parser.parse_args()


class DSDataset(data.Dataset):
    def __init__(self, data_dir, split_dir, fold, data_type='train', device='cuda:1', merge=False, twocls=False):
        # csv_file = os.path.join(split_dir, f'fold_{fold}.csv')
        # pd_data = pd.read_csv(csv_file)
        self.root_dir = data_dir
        # self.data_list = pd_data[data_type].dropna(axis=0,how='any').tolist()
        self.data_list = os.listdir(self.root_dir)
        self.device = device
        self.merge = merge
        if merge:
            # self.label_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
            self.label_dict = {1: 0, 2: 1, 3: 1, 4: 1}
        if twocls:
            tcls_data = []
            for item in self.data_list:
                # if item.split('-')[-1][0] == '0' or item.split('-')[-1][0] == '1':
                if item.split('-')[-1][0] != '0':
                    tcls_data.append(item)
            self.data_list = tcls_data



    def _get_h5(self, h5):
        f = h5py.File(h5, 'r')
        feature = f['feature'][:]
        feature = torch.from_numpy(feature)
        f.close()

        return feature

    def __getitem__(self, item):
        slide_name = self.data_list[item]
        data_path = os.path.join(self.root_dir,slide_name)
        data = self._get_h5(data_path)
        label = int(os.path.basename(slide_name).split('-')[-1][0])
        if self.merge:
            label = self.label_dict[label]
        return data, label

    def __len__(self):
        return len(self.data_list)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, (slide, label) in enumerate(train_loader):
        slide = slide.to(device)
        label = label.to(device)

        ins_prediction, bag_prediction, A, _ = model(slide)
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), label)
        max_loss = criterion(max_prediction.view(1, -1), label)
        loss = 0.5 * bag_loss + 0.5 * max_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] loss {}'.format(epoch, round(total_loss.item(), 3))

    return total_loss.item()


@torch.no_grad()
def val_one_epoch(model, val_loader, device, average=False, data_type='val'):
    model.eval()
    test_labels = torch.FloatTensor(len(val_loader.dataset))
    test_preds = torch.Tensor()
    # test_preds = torch.FloatTensor(len(val_loader.dataset))
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=75, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=75, colour='green')

    for i, (slide, label) in enumerate(val_loader):
        slide = slide.to(device)
        label = label.to(device)

        ins_prediction, bag_prediction, _, _ = model(slide)
        max_prediction, _ = torch.max(ins_prediction, 0)

        test_labels[i: i + slide.size(0)] = label.detach()[:].clone()

        if average:
            pred = (0.5 * F.softmax(max_prediction, dim=0) + 0.5 * F.softmax(bag_prediction, dim=0)).cpu()
        else:
            pred = F.softmax(bag_prediction, dim=1).cpu()

        test_preds = torch.concat((test_preds, pred), dim=0)
    
    return test_preds, test_labels
        

def cal_error(tile_max, ground_truth):
    prediction = [np.argmax(x) for x in tile_max]
    prediction = np.array(prediction)
    # labels = [np.argmax(x) for x in ground_truth]
    labels = np.array(ground_truth)
    cm = confusion_matrix(labels, prediction)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)  # 每一类的准确率
    overall_accuracy = accuracy_score(labels, prediction)

    # auc = roc_auc_score(labels, tile_max, average='macro', multi_class='ovr')
    auc = roc_auc_score(labels, prediction)
    # class_name = ['class 0 ','class 1', 'class 2', 'class 3', 'class 4']
    # class_name = ['class 0 ', 'class 1', 'class 2']
    class_name = ['class 0 ', 'class 1']
    report = classification_report(labels, prediction, digits=4, target_names=class_name)

    # 计算 F1 分数
    f1 = f1_score(labels, prediction, average='macro')

    return class_accuracies, overall_accuracy, auc, f1, cm, report


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    split_dir = args.split_dir
    src_dir = args.src_dir
    continue_fold = args.continue_fold

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f'{args.output_dir}/summary.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['fold', 'test class acc', 'test overall acc', 'test auc', 'test f1'])

    with open(f'{args.output_dir}/test_matrix.txt', 'w') as f:
        print('test start', file=f)

    with open(f'{args.output_dir}/test_report.txt', 'w') as f:
        print('test start', file=f)

    clean = True
    for fold in range(continue_fold,args.fold):
        if clean:
            data_dir = os.path.join(src_dir,'fold_{}'.format(fold))
        else:
            data_dir = os.path.join(src_dir)

        best_score = 0.
        # train_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='train', device=device, merge=True, twocls=True)
        # val_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='val', device=device, merge=True, twocls=True)
        test_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='test', device=device, merge=True, twocls=True)

        print(f'Using fold {fold}')
        # print(f'train: {len(train_set)}')
        # print(f'valid: {len(val_set)}')
        print(f'test: {len(test_set)}')

        # train_loader = data.DataLoader(train_set, batch_size=1, num_workers=args.n_workers, shuffle=True)
        # val_loader = data.DataLoader(val_set, batch_size=1, num_workers=args.n_workers, shuffle=False)
        test_loader = data.DataLoader(test_set, batch_size=1, num_workers=args.n_workers, shuffle=False)

        i_classifier = FCLayer(in_size=args.embed_dim, out_size=args.n_classes)
        b_classifier = BClassifier(input_size=args.embed_dim, output_class=args.n_classes)
        model = MILNet(i_classifier, b_classifier).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0.000005)

        output_dir = os.path.join(args.output_dir, str(fold))
        os.makedirs(output_dir, exist_ok=True)

        # print(f"Start training for {args.epochs} epochs")
        #
        # with open(f'{output_dir}/results.csv', 'w') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['epoch', 'val class acc', 'val overall acc','val auc', 'val f1'])
        #
        # with open(f'{output_dir}/val_matrix.txt', 'w') as f:
        #     print('val start', file=f)
        #
        # for epoch in range(args.epochs):
        #     _ = train_one_epoch(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch + 1)
        #     scheduler.step()
        #
        #     preds, labels = val_one_epoch(model=model, val_loader=val_loader, device=device,data_type='val')
        #     val_acc, val_all_acc, val_auc, val_f1, val_mat, _ = cal_error(preds, labels)
        #     print('Val\t[Fold {}][epoch {}] acc:{}\tall_acc:{:.4f}\tauc:{:.4f}\tf1-score:{:.4f}'.format(fold + 1,epoch + 1, val_acc, val_all_acc, val_auc, val_f1))
        #     print('val matrix ......')
        #     print(val_mat)
        #     current_score = val_f1
        #
        #     if current_score >= best_score:
        #         best_score = current_score
        #         obj = {
        #             'fold': fold + 1,
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'best_acc': best_score,
        #             'optimizer': optimizer.state_dict()
        #         }
        #         torch.save(obj, os.path.join(output_dir, f'dsmil_fold{fold + 1}.pth'))
        #     print('Val\tbest_score:{:.4f}'.format(best_score))
        #
        #     with open(f'{output_dir}/val_matrix.txt', 'a') as f:
        #         print(epoch + 1, file=f)
        #         print(val_mat, file=f)
        #
        #     with open(f'{output_dir}/results.csv', 'a') as csvfile:
        #         csv_writer = csv.writer(csvfile)
        #         csv_writer.writerow([epoch+1, val_acc, val_all_acc, val_auc, val_f1])

        model_path = os.path.join(output_dir, f'dsmil_fold{fold + 1}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        print("model load successfully.\nTest starts.")
        test_preds, test_labels = val_one_epoch(model=model, val_loader=test_loader, device=device, data_type='test')
        test_acc, test_acc_all, test_auc, test_f1, test_mat, test_report = cal_error(test_preds, test_labels)
        print('Test\t class_acc:{}\toverall_acc:{}\tauc:{}\tf1-score:{}\t'.format(test_acc, test_acc_all, test_auc, test_f1))
        print('test matrix ......')
        print(test_mat)
        with open(f'{args.output_dir}/test_matrix.txt', 'a') as f:
            print(fold + 1, file=f)
            print(test_mat, file=f)

        with open(f'{args.output_dir}/test_report.txt', 'a') as f:
            print(fold + 1, file=f)
            print(test_report, file=f)

        with open(f'{args.output_dir}/summary.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([fold + 1, test_acc, test_acc_all, test_auc, test_f1])

if __name__ == '__main__':
    opt = parse()
    main(opt)

        


