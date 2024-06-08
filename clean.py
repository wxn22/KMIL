import os
import sys
import argparse
import csv
import h5py
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter

import torch
import time
from torchvision import models
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from models.MIL_models import ABMIL, Feat_Classifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix



class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=0, verbose=False):
        self.patience = patience
        self.warmup_epoch = warmup_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.model_path = model_path

    def __call__(self, epoch, val_loss, model, val_acc=None):
        flag = False
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc >= self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, status='acc')
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch > self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss'):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))

class M2Dataset(data.Dataset):
    def __init__(self, data_dir, split_dir, fold, data_type):
        # csv_file = os.path.join(split_dir, f'fold_{fold}.csv')
        # pd_data = pd.read_csv(csv_file)
        self.root_dir = data_dir
        self.data_list = os.listdir(self.root_dir)
        # self.data_list = pd_data[data_type].dropna(axis=0,how='any').tolist()

    def _get_h5(self, h5):
        f = h5py.File(h5, 'r')
        feature = f['feature'][:]
        feature = torch.from_numpy(feature)
        f.close()

        return feature

    def __getitem__(self, item):
        slide_name = self.data_list[item]
        slide_id = slide_name.split('.')[0]
        split_ = slide_id.split('-')
        h5name = split_[0] + '-' + split_[-1]
        data_path = os.path.join(self.root_dir,h5name+'.h5')
        feature = self._get_h5(data_path)
        label = int(slide_name.split('-')[-1][0])
        sample = {'slide_id': slide_id, 'feat': feature, 'target': label}

        return sample

    def __len__(self):
        return len(self.data_list)

class TrDataset(data.Dataset):
    def __init__(self, data_dir):
        self.root_dir = data_dir
        self.data_list = os.listdir(self.root_dir)

    def _get_h5(self, h5):
        f = h5py.File(h5, 'r')
        feature = f['feature'][:]
        feature = torch.from_numpy(feature)
        f.close()

        return feature

    def __getitem__(self, item):
        slide_name = self.data_list[item]
        slide_id = slide_name.split('.')[0]
        data_path = os.path.join(self.root_dir,slide_name)
        feature = self._get_h5(data_path)
        label = int(slide_name.split('-')[-1][0])
        sample = {'slide_id': slide_id, 'feat': feature, 'target': label}

        return sample

    def __len__(self):
        return len(self.data_list)

def parse():
    parser = argparse.ArgumentParser('Clean features for MIL')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train', action='store_true')

    # Model parameters
    parser.add_argument('--model', type=str, default='resnet',help='choose model [resnet,attention,gradcam]')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help="The dimension of instance-level representations")

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/data_sdd/zrl/exter_test',help='data path')
    parser.add_argument('--split_dir', type=str, default='/data_sdc/zrl/code/dtfd/split1', help='split path')
    parser.add_argument('--device', type=str, default='cuda:3', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--output_dir', default='/data_sdc/zrl/code/dtfd/result_res_abmil', help='path where to save, empty for no saving')

    return parser.parse_args()
def m2_train_epoch(m2_epoch, model, optimizer, loader, criterion, device, num_classes, model_suffix='ABMIL',
                   dropout_rate=0):
    model.train()
    attns = {}
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with tqdm(total=len(loader)) as pbar:
        for _, sample in enumerate(loader):
            optimizer.zero_grad()
            slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
            feat = feat.to(device)
            target = target.to(device)
            if 'ABMIL' in model_suffix:
                logit, attn = model(feat)
                loss = criterion(logit, target.long())
            elif 'CLAM' in model_suffix:
                bag_weight = 0.5
                logit, attn, instance_dict = model(feat, target, instance_eval=True)
                instance_loss = instance_dict['instance_loss']
                loss = bag_weight * criterion(logit, target.long()) + (1 - bag_weight) * instance_loss
            else:
                raise NotImplementedError

            # calculate metrics
            attns[slide_id[0]] = attn
            logits = torch.cat((logits, logit.detach().cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)
            acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)
            loss_all += loss.detach().item() * len(target)

            # loss backward
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            pbar.set_description('[Epoch:{}] lr:{:.4f}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                 .format(m2_epoch, lr, loss_all / len(targets), acc, roc_auc, f1))
            pbar.update(1)
    return loss_all / len(targets), acc, roc_auc, f1, attns


def m2_pred(model, loader, criterion, device, num_classes, model_suffix='ABMIL', status='Val'):
    model.eval()
    attns = {}
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                feat = feat.to(device)
                target = target.to(device)
                if 'ABMIL' in model_suffix:
                    logit, attn = model(feat)
                elif 'CLAM' in model_suffix:
                    logit, attn, _ = model(feat, target)
                else:
                    raise NotImplementedError

                # calculate metrics
                attns[slide_id[0]] = attn
                loss = criterion(logit, target.long())
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss.item() * len(target)
                acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)

                pbar.set_description('[{}] loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                     .format(status,loss_all / len(targets), acc, roc_auc, f1))
                pbar.update(1)
    acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
    print(mat)
    return loss_all / len(targets), acc, roc_auc, mat, attns, f1
def patch_pred(model, loader, device):
    model.eval()
    instance_probs = {}
    with torch.no_grad():
        for i, sample in enumerate(loader):
            slide_id, feat = sample['slide_id'], sample['feat']
            feat = feat.to(device)
            logit = model(feat)
            probs = F.softmax(logit, dim=1)
            instance_probs[slide_id[0]] = probs.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return instance_probs

def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='weighted')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat
    return acc, f1, roc_auc

def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure
def draw_metrics(ts_writer, name, num_class, loss, acc, auc, mat, f1):
    ts_writer.add_scalar("{}/loss".format(name), loss)
    ts_writer.add_scalar("{}/acc".format(name), acc)
    ts_writer.add_scalar("{}/auc".format(name), auc)
    ts_writer.add_scalar("{}/f1".format(name), f1)
    if mat is not None:
        ts_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class))
def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    split_dir = args.split_dir
    data_dir = args.data_dir
    folds = args.fold
    num_classes = args.n_classes
    out_dir = args.output_dir
    lr = 1e-3
    min_lr = 1e-4
    MIL_model = 'ABMIL'
    dropout_rate = 0
    M2_patience = 20
    train = True

    os.makedirs(out_dir,exist_ok=True)

    start = 0
    for fold in range(start,folds):
        train_dset = M2Dataset(data_dir, split_dir, fold, data_type='train')
        # ran_sampler = RandomSampler(data_source=train_dset,num_samples=200)
        train_loader = DataLoader(train_dset, batch_size=1, shuffle=False, num_workers=0)
        val_dset = M2Dataset(data_dir, split_dir, fold, data_type='val')
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
        test_dset = M2Dataset(data_dir, split_dir, 0, data_type='test')
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
        # src_dset = TrDataset(data_dir)
        # src_loader = DataLoader(src_dset, batch_size=1, shuffle=False, num_workers=0)
        #
        model = ABMIL(n_classes=num_classes).to(device)
        # resnet50 = models.resnet50(pretrained=True)
        patch_model = Feat_Classifier(n_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        patch_probs = patch_pred(patch_model, test_loader, device)

        model_dir = os.path.join(out_dir, 'ABMIL_model_{}.pth'.format(fold))
        if not os.path.exists(model_dir):
            early_stopping = EarlyStopping(model_path=model_dir, patience=M2_patience, verbose=True)
            for m2_epoch in range(args.epochs):
                m2_train_epoch(m2_epoch, model, optimizer, train_loader, criterion, device, num_classes,
                               MIL_model, dropout_rate=dropout_rate)
                loss, acc, _, _, _, _ = m2_pred(model, val_loader, criterion, device, num_classes, MIL_model)
                counter = early_stopping(m2_epoch, loss, model, acc)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    break
                # adjust learning rate
                if counter > 0 and counter % 7 == 0 and lr > min_lr:
                    lr = lr / 3 if lr / 3 >= min_lr else min_lr
                    for params in optimizer.param_groups:
                        params['lr'] = lr
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
        _, _, _, _, attns, _ = m2_pred(model, src_loader, criterion, device, num_classes, MIL_model, 'Val')
        patch_model.load_state_dict(torch.load(model_dir, map_location=device), strict=False)
        patch_probs = patch_pred(patch_model, src_loader, device)

        mode = 'instance'
        percente = 0.9
        if mode == 'instance':
            stard = patch_probs
            save_path = os.path.join('/data_sdd/wxn/wocleansex_clean_var_0.9_ex_4','fold_{}'.format(fold))
        elif mode == 'attention':
            stard = attns
            save_path = os.path.join('/data_sdc/zrl/lymphnode/featureX20/wocleansex_clean_attn_new','fold_{}'.format(fold))
        elif mode == 'mix':
            stard = attns
            save_path = os.path.join('/data_sdb/wxn/wocleansex_clean_mix', 'fold_{}'.format(fold))
        with tqdm(total=len(stard)) as pbar:
            for slide_id, item in stard.items():
                if mode == 'instance':
                    # label = predict[index]
                    probs = torch.from_numpy(item)
                    probs = torch.transpose(probs, 1, 0)
                    score = torch.var(probs,dim=0)
                    # score = probs[label]

                elif mode == 'attention':
                    score = torch.from_numpy(item).squeeze(0)
                    # score = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
                elif mode == 'mix':
                    attn = torch.from_numpy(item).squeeze(0)
                    prob = torch.from_numpy(patch_probs[slide_id])  #NXK
                    # prob = torch.transpose(prob, 1, 0)
                    _, ptopk_id = torch.topk(attn, k=int(min(10,len(attn))), dim=0)
                    select_prob = prob[ptopk_id]
                    labels = [torch.argmax(prob) for prob in select_prob]
                    cnt = Counter(labels)
                    slide_label= cnt.most_common(1)[0][0]
                    prob = torch.transpose(prob, 1, 0)
                    score = prob[slide_label]
                    # attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
                    # score = prob * attn
                    # score = prob[slide_label]

                slide_patch_num = len(score)
                K = int(percente * slide_patch_num)

                h5py_path = os.path.join(data_dir, slide_id + '.h5')
                f = h5py.File(h5py_path, 'r')
                feature = f['feature'][:]
                feature = torch.from_numpy(feature)
                f.close()

                _, ptopk_id = torch.topk(score, k=K, dim=0)
                # _, sort_id = torch.sort(score, dim=0)
                # ptopk_id = torch.cat([sort_id[:8*K], sort_id[9*K:]])
                ptopk_features = feature[ptopk_id.numpy()]


                os.makedirs(save_path, exist_ok=True)
                f = h5py.File(os.path.join(save_path, slide_id + '.h5'), 'w')
                embedding = f.create_dataset('feature', data=ptopk_features.numpy())
                f.close()

                # index += 1
                pbar.set_description('mode:{}'.format(mode))
                pbar.update(1)


if __name__ == '__main__':
    opt = parse()
    main(opt)