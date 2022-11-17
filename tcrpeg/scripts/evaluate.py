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
    parser.add_argument('--test_path',type=str,default='../data/pdf_test.csv',help='The test file for prob inference')
    parser.add_argument('--model_path',type=str,help='The path to the trained model')
    parser.add_argument('--zip_type',type=int,default=1,help='The sample data is in the compression format gzip. For the whole data, you should specify zip_type=0')
    parser.add_argument('--path_aa_emb',type=str,default='../data/embedding_32.txt',help='The path to the word2vec embeddings of AAs. If not provided, will train new embeddings based on the training seqs')
    parser.add_argument('--num_layers',type=int,default=3,help='Number of layers')
    parser.add_argument('--hidden_size',type=int,default=64,help='Hidden size')
    args = parser.parse_args()

    compression_type = 'gzip' if args.zip_type==1 else None
    model = TCRpeg(hidden_size=args.hidden_size,num_layers = args.num_layers,embedding_path=args.path_aa_emb)
    model.create_model(load=True,path=args.model_path)

    Plot = plotting()
    eva = evaluation(model=model)

    test_data = pd.read_csv(args.test_path,compression=compression_type)
    data = {'seq':test_data['seq'].values,'count':test_data['count'].values}
    r,p_data,p_infer = eva.eva_prob(path=data)
    print('The Pearson\'s correlation coefficient is: ',r)
    Plot.plot_prob(p_data,p_infer)

