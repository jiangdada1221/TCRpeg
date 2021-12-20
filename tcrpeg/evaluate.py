import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
from model import *      
import argparse
from utils import *
from tcrpeg import tcrpeg
import scipy.stats as stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--seqs_test_path',type=str)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--gens_path',type=str,default='none')
    parser.add_argument('--store_gens_fn',type=str,default='none.txt')
    parser.add_argument('--info',type=str,default='none')
    parser.add_argument('--num_to_gen',type=int,default=0)
    parser.add_argument('--bi_encode',type=int,default=0,help='only 1 can intrigue bi encoding')
    parser.add_argument('--dropout',type=float,default=0.2)
    parser.add_argument('--num_layers',type=int,default=1)
    parser.add_argument('--latent_size',type=int,default=30)
    parser.add_argument('--beta',type=float,default=0.1)
    parser.add_argument('--vj_model',type=int,default=0)
    args = parser.parse_args()
    vj_model = True if args.vj_model != 0 else False
    #ensure reproduction
    torch.manual_seed(0)
    np.random.seed(0)

class evaluation:
    def __init__(self,model,frac=1.0,path_to_test=None,vj_model=False):
        self.model = model
        if path_to_test is not None:
            if not vj_model:
                self.seqs_test = pd.read_csv(path_to_test,sep='\t').sample(frac=frac)['seq'].values
            else :
                seqs_test_ = pd.read_csv(path_to_test,sep='\t').sample(frac=frac)
                self.seqs_test = seqs_test_['seq'].values
                self.vs_test = seqs_test_['v'].values
                self.js_test = seqs_test_['j'].values
        self.vj = vj_model
    
    def generate(self,num,batch_size):
        if self.vj:
            return self.model.generate_tcrpeg_vj( num, batch_size)
        else :
            return self.model.generate_tcrpeg(num,batch_size)

    def eva_prob(self,path_to_pdf,whole=False):
        #if whole=True, will return the kl of the whole test set
        #sample 1e5 seqs for kl estimation
        if not whole:
            seq_data = pd.read_csv(path_to_pdf,compression='gzip').sample(n=int(1e5))
        else :
            seq_data = pd.read_csv(path_to_pdf,compression='gzip')

        c_data_ = seq_data['count'].values
        seqs_ = seq_data['seq'].values
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
                log_probs = self.model.sampling_decoder(seq_batch) #change here
                record[i*batch_size : end] = np.exp(log_probs)
        kl = kl_divergence(p_data,record)
        corr = stats.pearsonr(p_data,record)[0]
        print('kl divergence and Pearson correlation coefficient are : {} and {}'.format(str(round(kl,4)),str(round(corr,4))))
        return kl,corr,p_data,record

    # print('begin generating')
    # #assert len(seqs_gens) == seqs_test
    # length_dis(seqs_gens,seqs_test,args.info+'_len_dis')
    # aa_num_dis(seqs_gens,seqs_test,args.info+'_aaNum_dis')
    
    # # seqs_gens = [seq for seq in seqs_gens if len(seq) >= 8]
    # # seqs_test = [seq for seq in seqs_test if len(seq) >= 8]
    # aas_dis(seqs_gens,seqs_test,args.info+'_aas_dis')
    # # pos_dis(seqs_gens,seqs_test,args.info+'_pos_dis')
    # if vj_model:
    #     v_dis(vs_gens,vs_test,args.info+'_v_dis')
    #     j_dis(js_gens,js_test,args.info + '_j_dis')