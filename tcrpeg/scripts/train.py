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
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--path_train',type=str,help='The path to training file (.csv). Column name of CDR3 should be \'seq\'. If using v and j genes, their columns should be \'v\' and \'j\'')
    parser.add_argument('--epoch',type=int,default=20,help='Number of epochs')
    parser.add_argument('--num_layers',type=int,default=3,help='Number of layers')
    parser.add_argument('--hidden_size',type=int,default=64,help='Hidden size')
    parser.add_argument('--batch_size',type=int,default=1000,help='batch_size')
    parser.add_argument('--zip_type',type=int,default=1,help='The sample data is in the compression format gzip. For the whole data, you should specify zip_type=0')
    parser.add_argument('--learning_rate',type=float,default=1e-4,help='Learning rate')
    parser.add_argument('--path_aa_emb',type=str,default='../data/embedding_32.txt',help='The path to the word2vec embeddings of AAs. If not provided, will train new embeddings based on the training seqs')
    parser.add_argument('--store_path',type = str,default='../results/model.pth',help='The path to store the trained model')
    args = parser.parse_args()

    aa_emb_path = args.path_aa_emb
    isExist = os.path.exists(aa_emb_path)
    compression_type = 'gzip' if args.zip_type==1 else None
    train_file = pd.read_csv(args.path_train,compression=compression_type)
    assert 'seq' in train_file.columns, 'the column name of CDR3 shoule be seq'
    seqs = train_file['seq'].values 

    vj=False
    if len(train_file.columns) == 3:
        if train_file.columns[1] == 'v' and train_file.columns[2] == 'j':
            vj=True
            vs,js = train_file['v'].values, train_file['j'].values
            vj_data = [[seqs[i],vs[i],js[i]] for i in range(len(seqs))]
    if not isExist: #no aa emb provided, tarin a new one
        _ = word2vec(path=seqs,epochs=20,batch_size=1000,device='cuda:0',lr=0.0001,window_size=2,record_path='../data/aa_emb_32.txt')
        aa_emb_path = '../data/aa_emb_32.txt'    
    if vj:
        model = TCRpeg(hidden_size=args.hidden_size,num_layers = args.num_layers,load_data=True,embedding_path=aa_emb_path,path_train=vj_data,vj=True)
        model.create_model(vj=True)
        model.train_tcrpeg_vj(epochs=args.epoch,batch_size=args.batch_size,lr=args.learning_rate)
    else :
        model = TCRpeg(hidden_size=args.hidden_size,num_layers = args.num_layers,load_data=True,embedding_path=aa_emb_path,path_train=seqs)    
        model.create_model()    
        model.train_tcrpeg(epochs=args.epoch,batch_size=args.batch_size,lr=args.learning_rate)

    save_path = args.store_path
    if not save_path.endswith('.pth'):
        save_path = save_path + '.pth'
    model.save(save_path)






