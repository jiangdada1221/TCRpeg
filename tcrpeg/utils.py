import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

aas = 'ACDEFGHIKLMNPQRSTVWY'
aa2idx = {aas[i]:i for i in range(len(aas))}
idx2aa = {v: k for k, v in aa2idx.items()}
vs = ['TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2','TRBV11-3','TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 
        'TRBV16', 'TRBV18','TRBV19','TRBV2', 'TRBV20-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV3-1', 'TRBV30',
'TRBV4-1', 'TRBV4-2','TRBV4-3',
 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8','TRBV6-1', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6','TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3',
  'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
js = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3','TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6','TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-6', 'TRBJ2-7']

def kl_divergence(dis1,dis2,avoid_zeros=True):
    if avoid_zeros:
        dis2 = [d + 1e-20 for d in dis2] #to avoid zero division
        dis1 = [d + 1e-20 for d in dis1]
    return sum([dis1[i]*np.log(dis1[i]/dis2[i]) for i in range(len(dis1))])

def valid_seq(seq):
    if len(seq) <= 2:
        return False
    if seq[-1] != 'F' and seq[:-2] != 'YV':
        return False
    if seq[0] != 'C':
        return False
    return True

def length_dis(seqs1:list,seqs2 :list,fig_name:str):
    #assert len(seqs1) == len(seqs2), 'length of seqs1 and seqs2 should be equal'
    lens1 = [seq for seq in seqs1 if valid_seq(seq)]
    lens1,lens2 = [len(seq) for seq in seqs1],[len(seq) for seq in seqs2]
    len2count1,len2count2 = defaultdict(int),defaultdict(int)
    for i in range(len(lens1)):
        
        len2count1[lens1[i]] += 1
    for i in range(len(lens2)):
        len2count2[lens2[i]] += 1
    k1,k2 = list(len2count1.keys()),list(len2count2.keys())
    #x_axis = list(set(k1+k2))
    #x_axis.sort()
    x_axis = list(range(1,31))
    fre1,fre2 = [len2count1[k]/len(lens1) for k in x_axis],[len2count2[k]/len(lens2) for k in x_axis]
    assert len(fre1) == len(fre2)
    kl = kl_divergence(fre1,fre2)
    plt.figure()
    plt.plot(x_axis,fre1,x_axis,fre2)
    plt.legend(['fre1','fre2'])
    plt.title(str(kl))
    plt.savefig('results/pictures/'+fig_name + '.png')
    return kl

def aas_dis(seqs1:list,seqs2:list,fig_name:str):
    #assert len(seqs1) == len(seqs2), 'len1 should be equal to len2'
    aa2pos_dic1 = [defaultdict(int) for _ in range(20)]
    aa2pos_dic2 = [defaultdict(int) for _ in range(20)]
    for i in range(len(seqs1)):
        seq1 = seqs1[i]
        if len(seq1) <= 2 or seq1[0] != 'C' or (seq1[-1] != 'F' and seq1[:-2] != 'YV'):
            continue
        for j,aa in enumerate(seq1):
            #if aa == 'e' or 
            aa2pos_dic1[aa2idx[aa]][j+1] += 1
    for i in range(len(seqs2)):
        seq2 = seqs2[i]
        for j,aa in enumerate(seq2):
            aa2pos_dic2[aa2idx[aa]][j+1] += 1
    
    fig,axs = plt.subplots(4,5,figsize=(16,16))
    x_axis = list(range(1,31))
    for row in range(4):
        for col in range(5):
            index = row*5 + col
            d1,d2,aa = aa2pos_dic1[index],aa2pos_dic2[index],idx2aa[index]
            y1,y2 = [d1[k] for k in x_axis],[d2[k] for k in x_axis]
            sum1,sum2 = sum(y1),sum(y2)
            if sum1 == 0 or sum2 == 0:
                print(sum1)
                print(sum2)
                print(aa)
            y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
            kl = kl_divergence(y1,y2)
            axs[row,col].plot(x_axis,y1,x_axis,y2)
            axs[row,col].set_title(str(aa) +' kl: ' +str(kl))
            axs[row,col].legend(['seqs1','seqs2'])
    plt.savefig('results/pictures/'+fig_name + '.png')

def pos_dis(seqs1:list,seqs2:list,fig_name:str):
    #now is for pos2-10
    pos2aa_dic1 = [defaultdict(int) for _ in range(9)]
    pos2aa_dic2 = [defaultdict(int) for _ in range(9)]

    for i in range(len(seqs1)):
        seq1 = seqs1[i]
        if len(seq1) < 10:
            print(seq1)
            continue
        for j in range(9): #j: 0-8
            pos2aa_dic1[j][seq1[j+1]] += 1
    for i in range(len(seqs2)):
        seq2 = seqs2[i]
        if len(seq2) < 10:
            print(seq2)
            continue
        for j in range(9): #j: 0-8
            pos2aa_dic2[j][seq2[j+1]] += 1
    fig,axs = plt.subplots(3,3,figsize=(16,16))
    x_axis = list('ACDEFGHIKLMNPQRSTVWY')
    for row in range(3):
        for col in range(3):
            index = row*3 + col
            d1,d2,pos = pos2aa_dic1[index],pos2aa_dic2[index],index+2
            y1,y2 = [d1[k] for k in x_axis],[d2[k] for k in x_axis]
            sum1,sum2 = sum(y1),sum(y2)
            y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
            kl = kl_divergence(y1,y2)
            axs[row,col].plot(x_axis,y1,x_axis,y2)
            axs[row,col].set_title('Position: '+str(pos) +' kl: ' +str(kl))
            axs[row,col].legend(['seqs1','seqs2'])
            plt.savefig('results/pictures/'+fig_name + '.png')

def aa_num_dis(seqs1:list,seqs2:list,fig_name:str):
    aas_1 = defaultdict(int)
    aas_2 = defaultdict(int)
    for i in range(len(seqs1)):
        seq1 = seqs1[i]
        if len(seq1) <= 2 or seq1[0] != 'C' or (seq1[-1] != 'F' and seq1[:-2] != 'YV'):
            continue
        for j,aa in enumerate(seq1):
            #if aa == 'e' or 
            aas_1[aa] += 1
    for i in range(len(seqs2)):
        seq2 = seqs2[i]
        for j,aa in enumerate(seq2):
            aas_2[aa] += 1
    x_axis = list(aas)
    y1,y2 = [aas_1[aa] for aa in x_axis],[aas_2[aa] for aa in x_axis]
    sum1,sum2 = sum(y1),sum(y2)
    
    y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
    kl = kl_divergence(y1,y2)
    plt.figure()
    plt.plot(x_axis,y1,x_axis,y2)
    plt.legend(['fre1','fre2'])
    plt.title(str(kl))
    plt.savefig('results/pictures/'+fig_name + '.png')

def v_dis(seqs1,seqs2,fig_name:str):
    vs_1 = defaultdict(int)
    vs_2 = defaultdict(int)
    for i in range(len(seqs1)):
        vs_1[seqs1[i]] += 1
    for s in seqs2:
        vs_2[s] += 1
    x_axis = vs
    y1,y2 = [vs_1[x]/len(seqs1) for x in vs], [vs_2[x]/len(seqs2) for x in vs]
    kl = kl_divergence(y1,y2)
    plt.figure()
    plt.plot(x_axis,y1,'o',x_axis,y2,'o')
    #plt.plot(x_axis,y1,x_axis,y2)
    plt.legend(['fre1','fre2'])
    plt.title(str(kl))
    plt.savefig('results/pictures/'+fig_name + '.png')

def j_dis(seqs1,seqs2,fig_name:str):
    vs_1 = defaultdict(int)
    vs_2 = defaultdict(int)
    for i in range(len(seqs1)):
        vs_1[seqs1[i]] += 1
    for s in seqs2:
        vs_2[s] += 1
    x_axis = js
    y1,y2 = [vs_1[x]/len(seqs1) for x in js], [vs_2[x]/len(seqs2) for x in js]
    kl = kl_divergence(y1,y2)
    plt.figure()
    plt.plot(x_axis,y1,'o',x_axis,y2,'o')
    plt.legend(['fre1','fre2'])
    plt.title(str(kl))
    plt.savefig('results/pictures/'+fig_name + '.png')
