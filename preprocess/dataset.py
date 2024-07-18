import pandas as pd
import numpy as np
import random as rd
from tqdm import tqdm
import datetime

path="./data"

def get_data(flag):
    '''
    Modification of code based on https://github.com/lpworld/DASL
    '''
    #flag = {1: book2mv, 2: mv2book}
    mode = "sim" # sample top sim
    window = 5
    sample = 10
    print(mode)

    data1 = pd.read_csv(path+'/book_rating.csv')
    data2 = pd.read_csv(path+'/mv_rating.csv')
    item_1 = pd.read_csv(path+'/item_book.csv')
    item_2 = pd.read_csv(path+'/item_mv.csv')
    item2fb_1 = list(item_1['asin'])
    item2fb_2 = list(item_2['asin'])
    ent = pd.read_csv(path+"/final_ent.csv")['asinId'].values.tolist()[:len(item_1)+len(item_2)]
    item2ent_1 = {i:ent.index(str(item2fb_1[i])) for i in range(item_count_1)}
    def rating2click(rating):
        click=1.0
        return click
    data1['click'] = data1.rating.apply(rating2click)
    data2['click'] = data2.rating.apply(rating2click)
    user_id = pd.read_csv(path+"/user_list.csv")
    data1 = pd.merge(data1, user_id, on=['user'], how='left')
    data2 = pd.merge(data2, user_id, on=['user'], how='left')
    item_id_1 = item_1[['itemId','asin']]
    data1 = pd.merge(data1, item_id_1, on=['asin'], how='left')
    item_id_2 = item_2[['itemId','asin']]
    data2 = pd.merge(data2, item_id_2, on=['asin'], how='left')
    data1 = data1[['userId','itemId','click','time']]
    data2 = data2[['userId','itemId','click','time']]

    user_count = len(user_id)
    item_count_1 = len(item2fb_1)
    item_count_2 = len(item2fb_2)
    item_count = item_count_1+item_count_2
    print(user_count, item_count_1, item_count_2, item_count)
    item_list1 = list(range(item_count_1))
    item_list2 = list(range(item_count_2))

    if flag == 1: #book2mv
        print('book to mv')
        try:
            src_book = np.load(path+"/kgcn_gru_cross_cs_src_book_max.npy", allow_pickle=True).tolist()
            if mode=="sample":
                tgt_mv_train = np.load(path+f"/trainset_2_0.8_sample_{sample}.npy", allow_pickle=True).tolist()
                tgt_mv_test = np.load(path+f"/testset_2_0.8_sample_{sample}.npy", allow_pickle=True).tolist()
            elif mode=="top":
                tgt_mv_train = np.load(path+f"/trainset_2_0.8_top10.npy", allow_pickle=True).tolist()
                tgt_mv_test = np.load(path+f"/testset_2_0.8_top10.npy", allow_pickle=True).tolist()
            else:
                tgt_mv_train = np.load(path+f"/trainset_2_0.8_sim10.npy", allow_pickle=True).tolist()
                tgt_mv_test = np.load(path+f"/testset_2_0.8_sim10.npy", allow_pickle=True).tolist()
        except:
            try:
                source_click = np.load(path+"/train_source_click.npy", allow_pickle=True).tolist()
                target_click = np.load(path+"/train_target_click.npy", allow_pickle=True).tolist()
                userid = list(set(user_id['userId']))
            except:
                userid = list(set(user_id['userId']))
                source_click = {}
                target_click = {}
                for user in tqdm(userid):
                    user1 = data1.loc[data1['userId']==user]
                    user1 = user1.sort_values(['time'])
                    tmp1 = user1.values.tolist()
                    source_click[user]=[[int(i[1]),i[3]] for i in tmp1]
                    
                    user2 = data2.loc[data2['userId']==user]
                    user2 = user2.sort_values(['time'])
                    tmp2 = user2.values.tolist()
                    target_click[user]=[[int(i[1]),i[3]] for i in tmp2]
                np.save(path+"/train_source_click.npy", source_click)
                np.save(path+"/train_target_click.npy", target_click)
            try:
                src_book = np.load(path+"/src_book.npy", allow_pickle=True).tolist()
                tgt_mv_train = np.load(path+"/tgt_mv_train.npy", allow_pickle=True).tolist()
                tgt_mv_test = np.load(path+"/tgt_mv_test.npy", allow_pickle=True).tolist()
            except:
                src_book, tgt_mv = data_step1(data1=data1, data2=data2, mask=0, user_id=user_id, item_list1=item_list1, item_list2=item_list2, item2ent_1=item2ent_1, window=window)
                tgt_mv_train, tgt_mv_test = data_step2(source_click, target_click, item2ent_1, tgt_mv, 'mv', userid, 0, mode)
                np.save(path+f"/src_book.npy", src_book)
                np.save(path+"/tgt_mv_train.npy", tgt_mv_train)
                np.save(path+"/tgt_mv_test.npy", tgt_mv_test)
        return src_book, tgt_mv_train, tgt_mv_test, 0, user_count, item_count
    else: #mv2book
        print('mv to book')
        try:
            src_mv = np.load(path+"/kgcn_gru_cross_cs_src_mv_max_mask_2000.npy", allow_pickle=True).tolist()
            if mode=="sample":
                tgt_book_train = np.load(path+f"/trainset_3_0.8_sample_{sample}.npy", allow_pickle=True).tolist()
                tgt_book_test = np.load(path+f"/testset_3_0.8_sample_{sample}.npy", allow_pickle=True).tolist()
            elif mode=="top":
                tgt_book_train = np.load(path+f"/trainset_3_0.8_top10.npy", allow_pickle=True).tolist()
                tgt_book_test = np.load(path+f"/testset_3_0.8_top10.npy", allow_pickle=True).tolist()
            else:
                tgt_book_train = np.load(path+f"/trainset_3_0.8_sim10.npy", allow_pickle=True).tolist()
                tgt_book_test = np.load(path+f"/testset_3_0.8_sim10.npy", allow_pickle=True).tolist()
        except:
            try:
                source_click = np.load(path+"/train_source_click.npy", allow_pickle=True).tolist()
                target_click = np.load(path+"/train_target_click.npy", allow_pickle=True).tolist()
                userid = list(set(user_id['userId']))
            except:
                userid = list(set(user_id['userId']))
                source_click = {}
                target_click = {}
                for user in tqdm(userid):
                    user1 = data1.loc[data1['userId']==user]
                    user1 = user1.sort_values(['time'])
                    tmp1 = user1.values.tolist()
                    source_click[user]=[[int(i[1]),i[3]] for i in tmp1]
                    
                    user2 = data2.loc[data2['userId']==user]
                    user2 = user2.sort_values(['time'])
                    tmp2 = user2.values.tolist()
                    target_click[user]=[[int(i[1]),i[3]] for i in tmp2]
                np.save(path+"/train_source_click.npy", source_click)
                np.save(path+"/train_target_click.npy", target_click)
            try:
                src_mv = np.load(path+"/src_mv.npy", allow_pickle=True).tolist()
                tgt_book_train = np.load(path+"/tgt_book_train.npy", allow_pickle=True).tolist()
                tgt_book_test = np.load(path+"/tgt_book_test.npy", allow_pickle=True).tolist()
            except:
                src_mv, tgt_book = data_step1(data1=data2, data2=data1, mask=2000, user_id=user_id, item_list1=item_list2, item_list2=item_list1, item2ent_1=item2ent_1, window=window)
                tgt_book_train, tgt_book_test = data_step2(source_click, target_click, item2ent_1, tgt_book, 'book', userid, 2000, mode)
                np.save(path+f"/src_mv.npy", src_mv)
                np.save(path+"/tgt_book_train.npy", tgt_book_train)
                np.save(path+"/tgt_book_test.npy", tgt_book_test)
        return src_mv, tgt_book_train, tgt_book_test, 2000, user_count, item_count

def data_step1(data1, data2, mask, user_id, item_list1, item_list2, item2ent_1, window):
    '''
    Modification of code based on https://github.com/lpworld/DASL
    '''
    print("step 1")
    dataset1, dataset2 = [], {}
    userid = list(set(user_id['userId']))
    for ss in userid:
        dataset2[ss]=[]
    for user in tqdm(userid):
        user1 = data1.loc[data1['userId']==user]
        user1 = user1.sort_values(['time'])
        tmp1 = user1.values.tolist()
        pos1 = [int(i[1]) for i in tmp1]
        user2 = data2.loc[data2['userId']==user]
        user2 = user2.sort_values(['time'])
        tmp2 = user2.values.tolist()
        pos2 = [int(i[1]) for i in tmp2]
        negs_all2 = list(set(item_list2)-set(pos2))
        neg2 = rd.sample(negs_all2, min(len(negs_all2), len(pos2)))
        len1 = len(tmp1)
        len2 = len(tmp2)
        if len1<2:
            continue
        elif len1<=window:
            for i in range(1,len1):
                if mask>0:
                    list1 = [int(j[1]) for j in tmp1[:i]]
                else:
                    list1 = [item2ent_1[int(j[1])] for j in tmp1[:i]]
                for m in range(window-i):
                    list1.append(mask)
                if mask>0:
                    dataset1.append([user, list1, int(tmp1[i][1]), i, 1])
                else:
                    dataset1.append([user, list1, item2ent_1[int(tmp1[i][1])], i, 1])
                neg = rd.choice(item_list1)
                while neg in pos1:
                    neg = rd.choice(item_list1)
                if mask>0:
                    dataset1.append([user, list1, neg, i, 0])
                else:
                    dataset1.append([user, list1, item2ent_1[neg], i, 0])
        else:
            for i in range(1,window):
                if mask>0:
                    list1 = [int(j[1]) for j in tmp1[:i]]
                else:
                    list1 = [item2ent_1[int(j[1])] for j in tmp1[:i]]
                for m in range(window-i):
                    list1.append(mask)
                if mask>0:
                    dataset1.append([user, list1, int(tmp1[i][1]), i, 1])
                else:
                    dataset1.append([user, list1, item2ent_1[int(tmp1[i][1])], i, 1])
                neg = rd.choice(item_list1)
                while neg in pos1:
                    neg = rd.choice(item_list1)
                if mask>0:
                    dataset1.append([user, list1, neg, i, 0])
                else:
                    dataset1.append([user, list1, item2ent_1[neg], i, 0])

            for i in range(window,len1):
                if mask>0:
                    list1 = [int(j[1]) for j in tmp1[i-window:i]]
                    dataset1.append([user, list1, int(tmp1[i][1]), window, 1])
                else:
                    list1 = [item2ent_1[int(j[1])] for j in tmp1[i-window:i]]
                    dataset1.append([user, list1, item2ent_1[int(tmp1[i][1])], window, 1])
                neg = rd.choice(item_list1)
                while neg in pos1:
                    neg = rd.choice(item_list1)
                if mask>0:
                    dataset1.append([user, list1, neg, window, 0])
                else:
                    dataset1.append([user, list1, item2ent_1[neg], window, 0])
        if mask>0:
            pos1 = [i for i in pos1]
            pos2 = [item2ent_1[i] for i in pos2]
            neg2 = [item2ent_1[i] for i in neg2]
        else:
            pos1 = [item2ent_1[i] for i in pos1]
            pos2 = [i for i in pos2]
            neg2 = [i for i in neg2]
        len_pos = len(pos1)
        dataset2[user].extend([[user, pos1, j, len_pos, 1] for j in pos2])
        dataset2[user].extend([[user, pos1, j, len_pos, 0] for j in neg2])
    return dataset1, dataset2

def data_step2(source_click, target_click, item2ent_1, dataset2, flag, userid, mask, mode='top'):
    print("step 2")
    tgt={}
    window = 10
    if mode=='top':
        if flag=='mv':
            for i,j in tqdm(source_click.items()):
                if len(j)>1:
                    tmp = []
                    targets = target_click[i]
                    seq = [item2ent_1[r[0]] for r in j]
                    if len(j)<=window:
                        seq.extend([mask]*(window-len(j)))
                        for k,t in targets:
                            tmp.append([i,seq,k,len(j),1])
                    else:
                        s_times = [r[1] for r in j]
                        idx = 0
                        tt = s_times[0]
                        for k,t in targets:
                            if idx!=len(seq):
                                while tt<t:
                                    idx+=1
                                    if idx!=len(seq):
                                        tt=s_times[idx]
                                    else:
                                        break
                            if idx>window:
                                tmp.append([i,seq[idx-window:idx],k,window,1])
                            else:
                                tmp.append([i,seq[:window],k,window,1])
                    tgt[i]=tmp
                else:
                    tgt[i]=[]
                    ''' use data that only have one source interaction
                    tmp = []
                    targets = target_click[i]
                    seq = [item2ent_1[r[0]] for r in j]
                    seq.extend([mask]*(window-len(j)))
                    for k,t in targets:
                        tmp.append([i,seq,k,len(j),1])
                    tgt[i] = tmp
                    '''
        else:
            for i,j in tqdm(target_click.items()):
                if len(j)>1:
                    tmp = []
                    targets = source_click[i]
                    seq = [r[0] for r in j]
                    if len(j)<=window:
                        seq.extend([mask]*(window-len(j)))
                        for k,t in targets:
                            tmp.append([i,seq,item2ent_1[k],len(j),1])
                    else:
                        s_times = [r[1] for r in j]
                        idx = 0
                        tt = s_times[0]
                        for k,t in targets:
                            if idx!=len(seq):
                                while tt<t:
                                    idx+=1
                                    if idx!=len(seq):
                                        tt=s_times[idx]
                                    else:
                                        break
                            if idx>window:
                                tmp.append([i,seq[idx-window:idx],item2ent_1[k],window,1])
                            else:
                                tmp.append([i,seq[:window],item2ent_1[k],window,1])
                    tgt[i]=tmp
                else:
                    tgt[i]=[]
                    '''use data that only have one source interaction
                    tmp = []
                    targets = source_click[i]
                    seq = [r[0] for r in j]
                    seq.extend([mask]*(window-len(j)))
                    for k,t in targets:
                        tmp.append([i,seq,item2ent_1[k],len(j),1])
                    tgt[i] = tmp
                    '''
    elif mode=='sim':
        def time_format(t1):
            t1 = t1.replace("-","")
            if t1[0]==" ":
                t1 = t1.strip()
            else:
                t1 = t1.split(" ")[0]
            return t1
        def time_delta(t1, t2):
            t1 = time_format(t1)
            t2 = time_format(t2)
            t1 = datetime.datetime.strptime(t1,'%Y%m%d')
            t2 = datetime.datetime.strptime(t2,'%Y%m%d')
            return abs((t1-t2).days)
        def sim_time(t1, t2, t):
            return time_delta(t,t1)<time_delta(t,t2)
        if flag=='mv':
            for i,j in tqdm(source_click.items()):
                if len(j)>1:
                    tmp = []
                    targets = target_click[i]
                    seq = [item2ent_1[r[0]] for r in j]
                    if len(j)<=window:
                        seq.extend([mask]*(window-len(j)))
                        for k,t in targets:
                            tmp.append([i,seq,k,len(j),1])
                    else:
                        s_times = [r[1].strip() for r in j]
                        idx = 0
                        tt = s_times[0]
                        for k,t in targets:
                            if idx!=len(seq):
                                while tt<=t.strip() or (idx>=window and sim_time(tt, s_times[idx-window], t.strip())):
                                    idx+=1
                                    if idx!=len(seq):
                                        tt=s_times[idx]
                                    else:
                                        break
                            if idx>window:
                                tmp.append([i,seq[idx-window:idx],k,window,1])
                            else:
                                tmp.append([i,seq[:window],k,window,1])
                    tgt[i]=tmp
                else:
                    tgt[i]=[]
                    '''use data that only have one source interaction
                    tmp = []
                    targets = target_click[i]
                    seq = [item2ent_1[r[0]] for r in j]
                    seq.extend([mask]*(window-len(j)))
                    for k,t in targets:
                        tmp.append([i,seq,k,len(j),1])
                    tgt[i] = tmp
                    '''
        else:
            for i,j in tqdm(target_click.items()):
                if len(j)>1:
                    tmp = []
                    targets = source_click[i]
                    seq = [r[0] for r in j]
                    if len(j)<=window:
                        seq.extend([mask]*(window-len(j)))
                        for k,t in targets:
                            tmp.append([i,seq,item2ent_1[k],len(j),1])
                    else:
                        s_times = [r[1].strip() for r in j]
                        idx = 0
                        tt = s_times[0]
                        for k,t in targets:
                            if idx!=len(seq):
                                while tt<=t.strip() or (idx>=window and sim_time(tt, s_times[idx-window], t.strip())):
                                    idx+=1
                                    if idx!=len(seq):
                                        tt=s_times[idx]
                                    else:
                                        break
                            if idx>window:
                                tmp.append([i,seq[idx-window:idx],item2ent_1[k],window,1])
                            else:
                                tmp.append([i,seq[:window],item2ent_1[k],window,1])
                    tgt[i]=tmp
                else:
                    tgt[i]=[]
                    '''use data that only have one source interaction
                    tmp = []
                    targets = source_click[i]
                    seq = [r[0] for r in j]
                    seq.extend([mask]*(window-len(j)))
                    for k,t in targets:
                        tmp.append([i,seq,item2ent_1[k],len(j),1])
                    tgt[i]=tmp
                    '''
    else:
        print("mode error")
        exit(0)
    for i,j in dataset2.items():
        if j!=[] and tgt[i]!=[]:
            cnt = 0
            neg = []
            for k in j:
                if cnt==len(tgt[i]):
                    break
                if k[-1]==0:
                    tt = tgt[i][cnt]
                    neg.append([i,tt[1],k[2],tt[3],0])
                    cnt+=1
            tgt[i].extend(neg)
    try:
        train_user = np.load(path+f"/trainset_tgt_{flag}.npy", allow_pickle=True).tolist()
        test_user = np.load(path+f"/testset_tgt_{flag}.npy", allow_pickle=True).tolist()
    except:
        train_user = rd.sample(dataset2.keys(), int(len(dataset2.keys())*0.8))
        test_user = list(set(dataset2.keys())-set(train_user))
        '''
        train_user = rd.sample(userid, int(len(userid)*0.8))
        test_user = list(set(userid)-set(train_user))
        '''
        np.save(path+f"/trainset_tgt_{flag}.npy", train_user)
        np.save(path+f"/testset_tgt_{flag}.npy", test_user)
    print(len(train_user), len(test_user))
    trainset = []
    testset = []
    for i in train_user:
        trainset.extend(tgt[i])
    for i in test_user:
        testset.extend(tgt[i])
    return trainset, testset
