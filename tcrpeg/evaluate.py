import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
from tcrpeg.model import *      
import argparse
from tcrpeg.utils import *
#from TCRpeg import tcrpeg
import scipy.stats as stats
from tqdm import tqdm

class evaluation:
    def __init__(self,model,vj_model=False):
        '''
        Used for evaluate the probability inference
        @model: the trained TCRpeg model
        @vj_model: whether the model is TCRpeg or TCRpeg_vj
        '''

        self.model = model
        # if path_to_test is not None:
        #     if not vj_model:
        #         self.seqs_test = pd.read_csv(path_to_test,sep='\t').sample(frac=frac)['seq'].values
        #     else :
        #         seqs_test_ = pd.read_csv(path_to_test,sep='\t').sample(frac=frac)
        #         self.seqs_test = seqs_test_['seq'].values
        #         self.vs_test = seqs_test_['v'].values
        #         self.js_test = seqs_test_['j'].values
        self.vj = vj_model
    
    def generate(self,num,batch_size):
        '''
        Generate some new Seqs
        @num: number of generated seqs
        @batch_size: batch_size used in generation

        @return: a list of seqs
        '''
        if self.vj:
            return self.model.generate_tcrpeg_vj( num, batch_size)
        else :
            return self.model.generate_tcrpeg(num,batch_size)

    def eva_prob(self,path,whole=False):
        #if whole=True, will return the r of the whole test set
        #sample 1e6 seqs for r estimation
        #path can be a csv file or a dict with keys of 'seq' and 'count'
        if isinstance(path,dict):
            seq_data = path
        else :
            if not whole:
                seq_data = pd.read_csv(path,compression='gzip').sample(n=int(1e6))
            else :
                seq_data = pd.read_csv(path,compression='gzip')

        c_data_ = list(seq_data['count'])
        seqs_ = list(seq_data['seq'])
        c_data,seqs = [],[]

        for i in range(len(seqs_)):  #only need seqs that has appearance > 2
            if c_data_[i] > 2:
                c_data.append(c_data_[i])
                seqs.append(seqs_[i])
        p_data = np.array(c_data)
        sum_p = np.sum(p_data)
        p_data = p_data / sum_p #normalized probability

        batch_size = 2000

        record = np.zeros(len(seqs))
        with torch.no_grad():        
            for i in tqdm(range(int(len(seqs)/batch_size)+1)):
                end = len(seqs) if (i+1) * batch_size > len(seqs) else (i+1) * batch_size
                seq_batch = seqs[i * batch_size : end]                
                log_probs = self.model.sampling_tcrpeg(seq_batch) #change here
                record[i*batch_size : end] = np.exp(log_probs)
        record_sum = np.sum(record)
        record = record/record_sum
        # kl = kl_divergence(p_data,record)
        corr = stats.pearsonr(p_data,record)[0]
        print('Pearson correlation coefficient are : {}'.format(str(round(corr,4))))
        return corr,p_data,record
