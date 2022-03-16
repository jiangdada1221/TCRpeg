from tcrpeg.TCRpeg import TCRpeg
from tcrpeg.classification import classification
import pandas as pd
import numpy as np
import os
from evaluate import evaluation
from tcrpeg.utils import plotting
import matplotlib.pyplot as plt
from collections import defaultdict
from tcrpeg.word2vec import word2vec



data = pd.read_csv('data/TCRs.csv')
tcrs,count = data['seq'].values,data['count'].values

word2vec(tcrs,30,10,'cuda:0',lr=0.0001,window_size=2)

model = TCRpeg(hidden_size=64,num_layers = 3,load_data=True,embedding_path='data/embedding_32.txt',path_train=tcrs)
model.create_model()

model.train_tcrpeg(30,64,1e-3)

Plot = plotting()
eva = evaluation(model=model)

data = {'seq':tcrs,'count':count} #here
_,p_data,p_infer = eva.eva_prob(data)
#Plot.plot_prob(p_data,p_infer)

# #gen
# gens = model.generate_tcrpeg(10000,1000)
# Plot.Length_Dis(tcrs,gens)
# Plot.AAs_Dis(tcrs,gens)

#get_embedding
embs = model.get_embedding(tcrs[:10])

#classification
data = pd.read_csv('data/classification.csv')
x,y = list(data['seq']),list(data['label'])
tcrpeg_c = classification(model,64*3,dropout=0.4)
tcrpeg_c.train(x,y,30,8,1e-3,val_split=0.2)
auc,aup,y_pres,y_trues = tcrpeg_c.evaluate(x,y,100)
