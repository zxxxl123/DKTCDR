import os, sys, time, random

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import tensorflow as tf
import collections
import random as rd
from tqdm import tqdm

from utils import DataInput, compute_auc_single
from preprocess.dataset import get_data
from preprocess.kg import *
from model import Model

flag_text = "all"
path="./data"
src2tgt = 1 # 1: book2mv, 2: mv2book
if src2tgt==1:
    task = 'book2mv'
else:
    task = 'mv2book'
day = 'test'
window = 5 # history sequence length
epoch = 150
dropout = False
hops = 2 # hops of sampling neighbors on knowledge graph
kge_dim = 64 # knowledge graph embedding dims
source_head = 4
source_block = 1
target_head = 4
target_block = 1
sample = 100
mode = "sim" # sample top sim 
agg = "sum" # sum concat neighbor
st = "1" # 0 1 2 3
neighbors = 16
ratio=0.8
csv_path = f"../result_{day}/{task}_st{st}_n{neighbors}_{hops}_{flag_text}_{agg}_{ratio*10}_{source_head}_{source_block}_{target_head}_{target_block}_{kge_dim}_{day}_{window}_allepoch{epoch}"
print(csv_path)
tf.reset_default_graph()
print('Data Loading')
t = time.time()
trainset1, trainset2, testset2, mask_id, user_count, item_count = get_data(src2tgt)

batch_size = 128*10
lr = 1e-1
pars = {"neighbor_sample_size": neighbors, "kge_dim": kge_dim, "n_iter": hops, 'aggregator': agg, 'window': window, 'mask_id':mask_id, 'source_head':source_head, 'source_block': source_block, 'target_head':target_head, 'target_block': target_block, "st": st}
if mode=="sample":
    maxlen=sample
else:
    maxlen=10
pars['maxlen'] = maxlen
random.shuffle(trainset1)
trainset1 = trainset1[:]
print("load done.")

random.shuffle(trainset2)
random.shuffle(testset2)
print(len(trainset1),len(trainset2),len(testset2))
print('dataset split done.') 

print("=================")
print(pars)
print("=================")
kg, pars['n_entities'], pars['n_relations'], pars['adj_entity'], pars['adj_relation'], adj_1, adj_2, pars['adj_1_entity'], pars['adj_1_relation'], pars['adj_2_entity'], pars['adj_2_relation'] = load_kg(pars)
doc_vec = np.load(path+f"/vectors.npy",allow_pickle=True)
pars['sc'] = doc_vec
print('all data load done, use time:%.1fs' % (time.time()-t))
############################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
# gpu_options = tf.GPUOptions(allow_growth=True)
for iters in range(5):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = Model(user_count, item_count, batch_size, pars, window)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        start_time = time.time()
        last_auc = 0.0
        saver = tf.train.Saver(max_to_keep=3)

        total_parameters = 0
        for variable in tf.trainable_variables():
            local_parameters=1
            shape = variable.get_shape()
            for i in shape:
                local_parameters*=i.value
                total_parameters+=local_parameters
        print("total parameters:",total_parameters)

        #pretrain in source domain
        res_dasl = []
        last_iter_loss = float("inf")
        for _ in range(epoch):
            print('Epoch %d START.' % (model.global_epoch_step.eval()))
            loss_All = base_loss_all = kg_loss_all = cl_loss_all = 0.0
            ''' update sampled neighbor per 10 epoch
            if (model.global_epoch_step.eval()+1) % 10 == 0:
                print("refresh adj")
                pars['adj_entity'], pars['adj_relation'] = construct_adj(pars, kg, pars['n_entities'])
                pars['adj_1_entity'], pars['adj_1_relation'], pars['adj_2_entity'], pars['adj_2_relation'] = construct_adj_cross_sample(pars['adj_entity'], pars['adj_relation'], adj_1, adj_2, pars, pars['n_entities'])
                model.update_adj(pars)
            '''
            for uij in tqdm(DataInput(trainset1, batch_size)):
                loss_all, base_loss, kg_loss, cl_lossi = model.train_1(sess, uij, lr, dropout)
                base_loss_all += base_loss
                loss_All += loss_all
                kg_loss_all += kg_loss
                cl_loss_all += cl_lossi
            print('Epoch %d \t[%.1fs]: train loss==[%.5f=%.5f+(1e-7)*%.5f+0.1*%.5f]' % (model.global_epoch_step.eval(), time.time()-start_time, loss_All, base_loss_all, kg_loss_all, cl_loss_all))
            res_dasl.append([model.global_epoch_step.eval(), loss_All, base_loss_all, kg_loss_all, cl_loss_all])
            sys.stdout.flush()
            file_name = csv_path+f"_{iters+1}_last_src.csv"
            columns = ['epoch', 'loss_All', 'base_loss_all', 'kg_loss_all', 'cl_loss_all']
            test = pd.DataFrame(columns=columns, data=res_dasl)
            test.to_csv(file_name, index=0)
            model.global_epoch_step_op.eval()
            if base_loss_all > last_iter_loss:
                break
            last_iter_loss = base_loss_all
        
        # train in target domain
        print("===================")
        print("train for domain 2")
        print("===================")
        model.global_epoch_step_zero.eval()
        res_dasl = []
        best_auc, best_rmse, best_logloss, best_mae = 0.0, 1.0, 1.0, 1.0
        for _ in range(epoch):
            print('Epoch %d START.' % (model.global_epoch_step.eval()))
            loss_All = base_loss_all = kg_loss_all = cl_loss_all = 0.0
            ''' update sampled neighbor per 10 epoch
            if (model.global_epoch_step.eval()+1) % 10 == 0:
                print("refresh adj")
                pars['adj_entity'], pars['adj_relation'] = construct_adj(pars, kg, pars['n_entities'])
                pars['adj_1_entity'], pars['adj_1_relation'], pars['adj_2_entity'], pars['adj_2_relation'] = construct_adj_cross_sample(pars['adj_entity'], pars['adj_relation'], adj_1, adj_2, pars, pars['n_entities'])
                model.update_adj(pars)
            '''
            for uij in tqdm(DataInput(trainset2, batch_size)):
                loss_all, base_loss, kg_loss, cl_lossu, cl_lossi= model.train_2(sess, uij, lr, dropout)
                loss_All += loss_all
                base_loss_all += base_loss
                kg_loss_all += kg_loss
                cl_loss_all += cl_lossu
                cl_loss_all += cl_lossi
            print("evaluating")
            test_b_auc, test_b_rmse, test_b_ll, mae_b = compute_auc_single(sess, model, DataInput(testset2, batch_size), 2)
            best_auc = max(best_auc, test_b_auc)
            best_rmse = min(best_rmse, test_b_rmse)
            best_logloss = min(best_logloss, test_b_ll)
            best_mae = min(best_mae, mae_b)
            print(f'Epoch {model.global_epoch_step.eval()} \tDomain B TEST_AUC RMSE MAE Logloss:{test_b_auc}, {test_b_rmse}, {mae_b}, {test_b_ll}')
            print('Epoch %d \t[%.1fs]: train loss==[%.5f=%.5f+(1e-7)*%.5f+0.1*%.5f]' % (model.global_epoch_step.eval(), time.time()-start_time, loss_All, base_loss_all, kg_loss_all, cl_loss_all))
            sys.stdout.flush()
            res_dasl.append([model.global_epoch_step.eval(), loss_All, base_loss_all, kg_loss_all, cl_loss_all, test_b_auc, test_b_rmse, test_b_ll, mae_b])
            sys.stdout.flush()
            file_name = csv_path+f"_{iters+1}_last_tgt.csv"
            columns = ['epoch', 'loss_All', 'base_loss_all', 'kg_loss_all', 'cl_loss_all', 'test_b_auc', 'test_b_rmse', 'test_b_ll', 'mae_b']
            test = pd.DataFrame(columns=columns, data=res_dasl)
            test.to_csv(file_name, index=0)
            model.global_epoch_step_op.eval()
        print(f"BEST RESULT\t AUC: {best_auc}, RMSE: {best_rmse}, MAE: {best_mae}, Logloss: {best_logloss}")
