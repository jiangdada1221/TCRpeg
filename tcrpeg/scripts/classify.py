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
    parser.add_argument('--dropout',type=float,default=0.2,help='The dropout rate')
    parser.add_argument('--path_train',type=str,help='The path to training file (.csv). Column name of CDR3 should be \'seq\' and the label should be \'label\'.')
    parser.add_argument('--path_test',type=str,help='The path to testing file (.csv). Column name of CDR3 should be \'seq\' and the label should be \'label\'.')
    parser.add_argument('--epoch',type=int,default=20,help='Number of epochs')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size')
    parser.add_argument('--num_layers',type=int,default=3,help='Number of layers')
    parser.add_argument('--hidden_size',type=int,default=512,help='Hidden size')
    parser.add_argument('--learning_rate',type=float,default=1e-4,help='Learning rate')
    parser.add_argument('--path_aa_emb',type=str,default='../data/embedding_32.txt',help='The path to the word2vec embeddings of AAs. If not provided, will train new embeddings based on the training seqs')
    parser.add_argument('--store_path',type = str,default='../results/model.pth',help='The path to store the trained tcrpeg-c model')
    args = parser.parse_args()

    aa_emb_path = args.path_aa_emb
    isExist = os.path.exists(aa_emb_path)
    train_file = pd.read_csv(args.path_train)
    assert 'seq' in train_file.columns and 'label' in train_file.columns, 'the column name of CDR3 shoule be seq'
    seqs_train = train_file['seq'].values 
    label_train = train_file['label'].values
    
    test_file = pd.read_csv(args.path_test)
    seqs_test = test_file['seq'].values 
    label_test = test_file['label'].values

    if not isExist: #no aa emb provided, tarin a new one
        _ = word2vec(path=seqs_train,epochs=20,batch_size=32,device='cuda:0',lr=0.0001,window_size=2,record_path='../data/aa_emb_32.txt')
        aa_emb_path = '../data/aa_emb_32.txt'

    #tcrpeg model, used to provide embeddings for tcrs
    model = TCRpeg(hidden_size=args.hidden_size,num_layers = args.num_layers,load_data=True,embedding_path=aa_emb_path,path_train=seqs_train)    
    model.create_model()    
    print('Train the TCRpeg model to embed the TCRs:')
    model.train_tcrpeg(epochs=20,batch_size=32,lr=1e-4)

    tcrpeg_c = classification(tcrpeg=model,embedding_size=args.hidden_size*args.num_layers)
    tcrpeg_c.train(x_train=seqs_train,y_train=label_train,epochs=args.epoch,batch_size=args.batch_size,lr=args.learning_rate)

    print('Begin evaluation on test set')
    auc,aup,y_pres,y_trues = tcrpeg_c.evaluate(x_test=seqs_test,y_test=label_test,batch_size=args.batch_size)
