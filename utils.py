import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, mean_absolute_error
from tqdm import tqdm

'''
Modification of code based on https://github.com/lpworld/DASL
'''

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data[:]
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i, lens = [], [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
            lens.append(t[4])
        return (u, hist, hist_cross, i, lens)
    
def cal_rmse(pred_score, label):
    mse = mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse
    
def compute_auc_single(sess, model, testset, domain):
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    arr_1 = []
    label_all_a = []
    score_all_a = []
    for uij_1 in tqdm(testset):
        if domain == 1:
            a = model.test_auc1(sess, uij_1)
        else:
            a = model.test_auc2(sess, uij_1)
        score, label, user = a
        for index in range(len(score)):
            label_all_a.append(label[index])
            score_all_a.append(score[index])
            if label[index] > 0:
                arr_1.append([0, 1, score[index]])
            elif label[index] == 0:
                arr_1.append([1, 0, score[index]])
    rmse_a = cal_rmse(score_all_a, label_all_a)
    mae_a = mean_absolute_error(label_all_a, score_all_a)
    score_all_a = [max(1e-15, p) for p in score_all_a]
    ll_a = log_loss(y_pred=score_all_a, y_true=label_all_a)
    arr_1 = sorted(arr_1, key=lambda d:d[2])
    auc_1 = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr_1:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc_1 += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_1) - 1e-3

    if tp2 > threshold or fp2 > threshold:
        auc_1 = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc_1 = (1.0 - auc_1 / (2.0 * tp2 * fp2))
    else:
        auc_1 = -0.5
    return auc_1, rmse_a, ll_a, mae_a