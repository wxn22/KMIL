import matplotlib.pyplot as plt
from dsmil_model import FCLayer, BClassifier, MILNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import glob
import os
import h5py
import csv
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, classification_report, RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sn
from pylab import mpl

def parse():

    parser = argparse.ArgumentParser(description='Train for dsmil')
    parser.add_argument('--epochs', type=int, default=100)

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=1024, help="The dimension of instance-level representations")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Dataset parameters
    parser.add_argument('--src_dir', type=str, default='/data_sdd/wxn/wocleansex_clean_var_0.9_ex_3', help='data path')
    parser.add_argument('--split_dir', type=str, default='/data_sdc/zrl/code/dtfd/split1', help='split path')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--continue_fold', type=int, default=0)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--output_dir', default='/data_sdb/wxn/dsmil/result_var_0.9_2cls+_ex', help='path where to save, empty for no saving')


    return parser.parse_args()


class DSDataset(data.Dataset):
    def __init__(self, data_dir, split_dir, fold, data_type='ex', device='cuda:1', merge=True, twocls=True):
        self.root_dir = data_dir
        if data_type == 'in':
            csv_file = os.path.join(split_dir, f'fold_{fold}.csv')
            pd_data = pd.read_csv(csv_file)
            self.data_list = pd_data['test'].dropna(axis=0, how='any').tolist()
        else:
            self.data_list = os.listdir(data_dir)
        self.device = device
        self.merge = merge
        if merge:
            self.label_dict = {1: 0, 2: 1, 3: 1, 4: 1}
            # self.label_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
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
        data_path = os.path.join(self.root_dir, slide_name)
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

        ins_prediction, bag_prediction, _, _ = model(slide)
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
def val_one_epoch(model, val_loader, device, average=False):
    model.eval()
    test_labels = torch.FloatTensor(len(val_loader.dataset))
    test_preds = torch.Tensor()
    # test_preds = torch.FloatTensor(len(val_loader.dataset))
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


def cal_error(tile_max, ground_truth, outputdir):
    prediction = [np.argmax(x) for x in tile_max]
    prediction = np.array(prediction)
    # print(prediction)

    labels = [np.argmax(x) for x in ground_truth]
    labels = np.array(ground_truth)

    cm = confusion_matrix(labels, prediction)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)  # 每一类的准确率
    overall_accuracy = accuracy_score(labels, prediction)
    class_name = ['class 0 ', 'class 1','class 2']
    report = classification_report(labels, prediction, digits=4, target_names=class_name)


    # 计算 F1 分数
    f1 = f1_score(labels, prediction, average='macro')

    return class_accuracies, overall_accuracy, auc, f1, cm, report

def confusion_vis(C,output):
    #
    # labels = ['non-tumor', 'lymphoma','breast','squamous', 'thyroid']
    labels = ['non-tumor', 'lymphoma', 'metastasis']
    # labels = ['non-tumor','lymphoma']
    # labels = ['lymphoma', 'metastasis']
    x_tick = labels
    y_tick = labels

    _data = {}

    for IDX in range(len(x_tick)):
        _data[x_tick[IDX]] = C[IDX]

    pd_data = pd.DataFrame(C, index=y_tick, columns=x_tick)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    font = {'family': 'sans-serif',
            'color': 'k',
            'weight': 'normal',
            'size': 20}
    _f, ax = plt.subplots(figsize=(12, 10))
    ax = sn.heatmap(pd_data, annot=True, ax=ax, fmt='.1f', cmap=plt.cm.Blues, annot_kws={"size": 32})

    plt.xlabel('Prediction', fontsize=20, color='k')
    plt.ylabel('Truth', fontsize=20, color='k')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('confusion matrix', fontsize=20)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)

    cbar = ax.collections[0].colorbar
    # cbar.set_label(r'$MNI$', fontdict=font)

    plt.savefig(os.path.join(output,'cm.png'))

def tsne_vis(X,y,path,feat_type):
    X = np.array(X)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    figure = plt.figure(figsize=(10, 8), dpi=300)

    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='.', c=y, cmap='tab10', alpha=0.8, s=100)
    # plt.text(0.5, 1, 'AUC on Test Dataset: {}'.format(auc), ha='center', va='center', fontsize=12,
    #          bbox=dict(facecolor='white', alpha=0.5))
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    save_path = os.path.join(path,feat_type+'_1_tsne.png')
    plt.savefig(save_path)

def plot_speci(in_preds,in_labels,ex_preds,ex_labels,output_dir):
    # in_prediction = [np.argmax(x) for x in in_preds]
    # ex_prediction = [np.argmax(x) for x in ex_preds]
    # in_prediction = np.array(in_prediction)
    # ex_prediction = np.array(ex_prediction)

    # in_labels = [np.argmax(x) for x in in_labels]
    # in_labels = np.array(in_labels)
    # ex_labels = np.array(ex_labels)
    in_y = []
    for y in in_labels:
        if y == 0:
            in_y.append([1,0])
        else:
            in_y.append([0,1])
    ex_y = []
    for y in ex_labels:
        if y == 0:
            ex_y.append([1, 0])
        else:
            ex_y.append([0, 1])
    # in_y = label_binarize(in_labels, classes=[0, 1])
    # ex_y = label_binarize(ex_labels,classes=[0, 1])
    in_fpr, in_tpr, _ = roc_curve(np.array(in_y).ravel(), np.array(in_preds).ravel())
    ex_fpr, ex_tpr, _ = roc_curve(np.array(ex_y).ravel(), np.array(ex_preds).ravel())
    in_roc_auc = auc(in_fpr, in_tpr)
    ex_roc_auc = auc(ex_fpr, ex_tpr)
    lw = 2
    plt.figure()
    plt.plot(in_fpr, in_tpr, label='test', color='blue', linewidth=4)
    plt.plot(ex_fpr, ex_tpr, label='external_test', color='red', linewidth=4)

    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir,"Micro-averaged.png"))

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


    # with open(f'{args.output_dir}/report.txt', 'w') as f:
    #     print('test start', file=f)
    # inter_dir = r'/data_sdb/wxn/wocleansex_clean_var_0.9'

    for fold in range(continue_fold, args.fold):
        data_dir = os.path.join(src_dir,'fold_{}'.format(fold))
        inter_dir = os.path.join(r'/data_sdb/wxn/wocleansex_clean_var_0.9','fold_{}'.format(fold))
        # data_dir = src_dir

        best_score = 0.
        # train_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='train', device=device)
        # val_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='val', device=device)
        in_set = DSDataset(data_dir=inter_dir, split_dir=split_dir, fold=fold, data_type='in', device=device)
        ex_set = DSDataset(data_dir=data_dir, split_dir=split_dir, fold=fold, data_type='ex', device=device)

        print(f'Using fold {fold}')
        # print(f'test: {len(test_set)}')

        # train_loader = data.DataLoader(train_set, batch_size=1, num_workers=args.n_workers, shuffle=False)
        # val_loader = data.DataLoader(val_set, batch_size=1, num_workers=args.n_workers, shuffle=False)
        in_loader = data.DataLoader(in_set, batch_size=1, num_workers=args.n_workers, shuffle=False)
        ex_loader = data.DataLoader(ex_set, batch_size=1, num_workers=args.n_workers, shuffle=False)

        i_classifier = FCLayer(in_size=args.embed_dim, out_size=args.n_classes)
        b_classifier = BClassifier(input_size=args.embed_dim, output_class=args.n_classes)
        model = MILNet(i_classifier, b_classifier).to(device)

        output_dir = os.path.join(args.output_dir, str(fold))
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'dsmil_fold{fold + 1}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        print("model load successfully.\nTest starts.")
        in_preds, in_labels = val_one_epoch(model=model, val_loader=in_loader, device=device)
        ex_preds, ex_labels = val_one_epoch(model=model, val_loader=ex_loader, device=device)
        # print(test_preds,test_labels)
        plot_speci(in_preds,in_labels,ex_preds,ex_labels,output_dir)
        # _, _, _, _, test_mat, report = cal_error(test_preds, test_labels, output_dir)
        # print('Test\t class_acc:{}\toverall_acc:{}\tauc:{}\tf1-score:{}\t'.format(test_acc, test_acc_all, test_auc, test_f1))
        # print('test matrix ......')
        # print(test_mat)

        # with open(f'{args.output_dir}/report.txt', 'a') as f:
        #     print(fold + 1, file=f)
        #     print(report, file=f)

        # confusion_vis(test_mat, output_dir)
        # tsne = True
        # if tsne == True:
        #     model.eval()
        #     feature_x = []
        #     label_y = []
        #     with torch.no_grad():
        #         with tqdm(total=len(train_loader)) as pbar:
        #             for _, (slide, label) in enumerate(train_loader):
        #                 feat = slide.to(device)
        #                 _, _, _, bag_feat = model(feat)
        #                 feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
        #                 label_y += label.numpy().tolist()
        #                 pbar.update(1)
        #         with tqdm(total=len(val_loader)) as pbar:
        #             for _, (slide, label) in enumerate(val_loader):
        #                 feat = slide.to(device)
        #                 _, _, _, bag_feat = model(feat)
        #                 feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
        #                 label_y += label.numpy().tolist()
        #                 pbar.update(1)
        #         with tqdm(total=len(test_loader)) as pbar:
        #             for _, (slide, label) in enumerate(test_loader):
        #                 feat = slide.to(device)
        #                 _, _, _, bag_feat = model(feat)
        #                 feature_x.append(bag_feat.cpu().squeeze(0).numpy().tolist())
        #                 label_y += label.numpy().tolist()
        #                 pbar.update(1)
        #     print(len(feature_x))
        #     print(len(label_y))
        #     tsne_vis(feature_x, label_y, output_dir,'5cls')


if __name__ == '__main__':
    opt = parse()
    main(opt)

