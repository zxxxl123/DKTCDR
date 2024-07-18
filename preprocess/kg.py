import numpy as np
import os
from tqdm import tqdm

'''
Modification of code based on https://github.com/hwwang55/KGCN
'''

path="../data"

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg
def construct_adj(pars, kg, entity_num):
    print('constructing adjacency matrix ...')
    adj_entity = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    adj_relation = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= pars["neighbor_sample_size"]:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation

def construct_adj_cross(pars, kg, entity_num, idx1, idx2, idx3):
    print('constructing adjacency cross matrix ...')
    adj_1, adj_2 = [], []
    domain2_1 = list(range(idx1))
    domain2_2 = list(range(idx2, entity_num))
    domain1 = list(range(idx1, entity_num))
    item1 = list(range(idx1, idx2))
    item2 = list(range(idx1))
    if os.path.exists(path+f"/adj_1.npy"):
        adj_1 = np.load(path+f"/adj_1.npy",allow_pickle=True).tolist()
    else:
        for entity in item2:
            adj_1.append([])

        for entity in tqdm(domain1):
            sampled_indices=[]
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if entity>idx2-1:
                if n_neighbors>idx3:
                    adj_1.append(sampled_indices)
                    continue
            tmp_iter1_cross = []
            tmp = [i[0] for i in neighbors]
            if list(set(tmp)&set(item2))==[]:
                for i in neighbors:
                    tmp_iter2 = kg[i[0]]
                    tmp_iter2.sort(key=lambda ele:ele[0])
                    tmp = [i[0] for i in tmp_iter2]
                    if list(set(tmp)&set(item2))==[]:
                        continue
                    else:
                        tmp_iter1_cross.append(i)
            else:
                for i in neighbors:
                    if i[0] not in item2:
                        tmp_iter2 = kg[i[0]]
                        tmp_iter2.sort(key=lambda ele:ele[0])
                        tmp = [i[0] for i in tmp_iter2]
                        if list(set(tmp)&set(item2))==[]:
                            continue
                        else:
                            tmp_iter1_cross.append(i)
                    if i[0] in item2:
                        tmp_iter1_cross.append(i)
            sampled_indices.extend(tmp_iter1_cross)
            adj_1.append(sampled_indices)
        np.save(path+f"/adj_1.npy",adj_1)
    if os.path.exists(path+f"/adj_2.npy"):
        adj_2 = np.load(path+f"/adj_2.npy",allow_pickle=True).tolist()
    else:
        for entity in tqdm(domain2_1):
            sampled_indices=[]
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors>idx3:
                adj_2.append(sampled_indices)
                continue
            tmp_iter1_cross = []
            tmp = [i[0] for i in neighbors]
            if list(set(tmp)&set(item1))==[]:
                for i in neighbors:
                    tmp_iter2 = kg[i[0]]
                    tmp_iter2.sort(key=lambda ele:ele[0])
                    tmp = [i[0] for i in tmp_iter2]
                    if list(set(tmp)&set(item1))==[]:
                        continue
                    else:
                        tmp_iter1_cross.append(i)
            else:
                for i in neighbors:
                    if i[0] not in item1:
                        tmp_iter2 = kg[i[0]]
                        tmp_iter2.sort(key=lambda ele:ele[0])
                        tmp = [i[0] for i in tmp_iter2]
                        if list(set(tmp)&set(item1))==[]:
                            continue
                        else:
                            tmp_iter1_cross.append(i)
                    else:
                        tmp_iter1_cross.append(i)
            sampled_indices.extend(tmp_iter1_cross)
            adj_2.append(sampled_indices)

        for entity in item1:
            adj_2.append([])

        for entity in tqdm(domain2_2):
            sampled_indices=[]
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors>idx3:
                adj_2.append(sampled_indices)
                continue
            tmp_iter1_cross = []
            tmp = [i[0] for i in neighbors]
            if list(set(tmp)&set(item1))==[]:
                for i in neighbors:
                    tmp_iter2 = kg[i[0]]
                    tmp_iter2.sort(key=lambda ele:ele[0])
                    tmp = [i[0] for i in tmp_iter2]
                    if list(set(tmp)&set(item1))==[]:
                        continue
                    else:
                        tmp_iter1_cross.append(i)
            else:
                for i in neighbors:
                    if i[0] not in item1:
                        tmp_iter2 = kg[i[0]]
                        tmp_iter2.sort(key=lambda ele:ele[0])
                        tmp = [i[0] for i in tmp_iter2]
                        if list(set(tmp)&set(item1))==[]:
                            continue
                        else:
                            tmp_iter1_cross.append(i)
                    else:
                        tmp_iter1_cross.append(i)
            sampled_indices.extend(tmp_iter1_cross)
            adj_2.append(sampled_indices)
        np.save(path+f"/adj_2.npy",adj_2)
    return adj_1, adj_2

def construct_adj_cross_sample(adj_ent, adj_rel, adj_1, adj_2, pars, entity_num):
    print('constructing adjacency cross matrix ...')
    adj_1_entity = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    adj_1_relation = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    adj_2_entity = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    adj_2_relation = np.zeros([entity_num, pars["neighbor_sample_size"]], dtype=np.int64)
    for idx,j in tqdm(enumerate(range(len(adj_1)))):
        neighbors = adj_1[idx]
        n_neighbors = len(neighbors)
        if n_neighbors==0:
            adj_1_entity[j] = adj_ent[j].copy()
            adj_1_relation[j] = adj_rel[j].copy()
            continue
        elif n_neighbors >= pars["neighbor_sample_size"]:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=True)
        adj_1_entity[j] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_1_relation[j] = np.array([neighbors[i][1] for i in sampled_indices])

    for idx,j in tqdm(enumerate(range(len(adj_2)))):
        neighbors = adj_2[idx]
        n_neighbors = len(neighbors)
        if n_neighbors==0:
            adj_2_entity[j] = adj_ent[j].copy()
            adj_2_relation[j] = adj_rel[j].copy()
            continue
        elif n_neighbors >= pars["neighbor_sample_size"]:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=pars["neighbor_sample_size"], replace=True)
        adj_2_entity[j] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_2_relation[j] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_1_entity, adj_1_relation, adj_2_entity, adj_2_relation

def load_kg(pars):
    print('reading KG file ...')
    kg_file = path+"/kg"
    idx1 = 1349
    idx2 = 2485
    idx3 = 1000

    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(pars, kg, n_entity)
    adj_1, adj_2 = construct_adj_cross(pars, kg, n_entity, idx1, idx2, idx3)
    adj_1_entity, adj_1_relation, adj_2_entity, adj_2_relation = construct_adj_cross_sample(adj_entity, adj_relation, adj_1, adj_2, pars, n_entity)

    return kg, n_entity, n_relation, adj_entity, adj_relation, adj_1, adj_2, adj_1_entity, adj_1_relation, adj_2_entity, adj_2_relation