from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.classification import classification
import pandas as pd
import numpy as np
import os
from tcrpeg.evaluate import evaluation
from tcrpeg.utils import plotting
from tcrpeg.word2vec import word2vec
import warnings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script of evaluating probability inference')    
    parser.add_argument('--model_path',type=str,help='The path to the trained model')
    parser.add_argument('--num_layers',type=int,default=3,help='Number of layers')
    parser.add_argument('--hidden_size',type=int,default=64,help='Hidden size')
    parser.add_argument('--vj',type=int,default=0, help='Whether to generate the corresponding v and j genes. If set to 1, will generate v and j genes and the model should be the tcrpeg_vj model')
    parser.add_argument('--n',type=int,default=10000, help='Number of generated seqs')
    parser.add_argument('--batch_num',type=int,default=1000, help='Generate the seqs batch-by-batch. Note that n (mod) batch_num should be 0')
    parser.add_argument('--store_path',type=str,help='Path to store the generated seqs')
    parser.add_argument('--path_aa_emb',type=str,default='../data/embedding_32.txt',help='The path to the word2vec embeddings of AAs. If not provided, will train new embeddings based on the training seqs')
    args = parser.parse_args()

    vj = True if args.vj == 1 else False
    model = TCRpeg(hidden_size=args.hidden_size,num_layers = args.num_layers,embedding_path=args.path_aa_emb,vj=vj)
    model.create_model(load=True,vj=vj,path=args.model_path)
    if vj:
        generated_data = model.generate_tcrpeg_vj(num_to_gen=args.n,batch_size=args.batch_num,record_path=args.store_path)
    else :
        generated_data = model.generate_tcrpeg(num_to_gen=args.n,batch_size=args.batch_num,record_path = args.store_path)
    
