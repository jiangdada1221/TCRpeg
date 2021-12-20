from model import FC_NN_large , FC_NN_medium, FC_NN_small
import torch              
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import roc_auc_score as AUC
import os

class classification:
    '''
    Simple FC models that utilize the embedding from trained tcrpeg model
    '''
    def __init__(self,tcrpeg,device,embedding_size,dropout=0.0,model_size='medium',last_layer=False):

        self.emb_model = tcrpeg
        self.device = device
        if model_size=='small':
            self.model = FC_NN_small(embedding_size,last_layer,device)
        elif model_size == 'medium':
            self.model = FC_NN_medium(embedding_size,dropout,last_layer,device,True)
        else :
            assert model_size == 'large'
            self.model = FC_NN_large(embedding_size,dropout,last_layer,device,True)
        self.model = self.model.to(device)

        self.criterion = nn.BCELoss()
        self.last_layer = last_layer
        self.device = device
    
    def load_model(self,path):
        #load the pre_trained model
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def train(self,epochs,batch_size,lr,save_name=None):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        infer = list(range(len(self.x_train)))
        x_train,y_train = self.x_train,self.y_train
        #self.evaluate(batch_size=batch_size,verbose=False)
        for epoch in range(epochs):
            self.model.train()
            print('begin epoch :',epoch+1)
            np.random.shuffle(infer)
            x_train,y_train = x_train[infer],y_train[infer]
            loss_ = []
            y_pres_whole = []
            for iter in tqdm(range(len(infer)//batch_size)):
                x_batch = x_train[iter*batch_size : (iter+1) * batch_size]
                y_batch = y_train[iter*batch_size : (iter+1) * batch_size]
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                y_pres = self.model(x_batch).view(batch_size)
                loss = self.criterion(y_pres,y_batch)
                optimizer.zero_grad()
                loss_.append(loss.item()*batch_size)
                loss.backward()
                optimizer.step()
                #y_pres_whole = y_pres_whole + list(y_pres.detach().cpu().numpy())
            # auc = AUC(x_test,)
            print('mean loss :',sum(loss_)/len(x_train))
            self.evaluate(batch_size=batch_size,verbose=False)
            if epoch % 30 == 0 and epoch != 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
        print('end training')
        self.evaluate(batch_size=batch_size,verbose=False)
        if save_name is not None:
            torch.save(self.model.state_dict(),'results/'+save_name + '.pth')

    def evaluate(self,batch_size,verbose=True):
        time_now = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
        y_pres_whole = []
        y_trues = []
        self.model.eval()
        with torch.no_grad():
            loss_ = []
            for iter in tqdm(range(int(len(self.x_test)/batch_size))):
                end = (iter+1) * batch_size if iter != int(len(self.x_test)/batch_size)-1 else len(self.x_test)
                x_batch = self.x_test[iter*batch_size : end]
                y_batch = self.y_test[iter*batch_size : end]
                y_trues += list(y_batch)
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                batch_size_temp = len(x_batch)
                y_pres = self.model(x_batch).view(batch_size_temp)
                loss_.append(self.criterion(y_pres,y_batch).item()*batch_size_temp)
                y_pres = y_pres.detach().cpu().numpy()
                y_pres_whole = y_pres_whole + list(y_pres)
            loss_avg = sum(loss_) / len(self.x_test)
            auc = AUC(y_trues,y_pres_whole)
            #auc = self.acc(y_trues,y_pres_whole)
            print('evaluation, avg_loss and auc is :{} and {}'.format(str(loss_avg),str(auc)))
        #sanity test
        # with open('checkpoints/sanity_3.txt','w') as f:
        #     for i in range(len(y_pres_whole)):
        #         f.write(str(y_pres_whole[i]) + ' ' + str(y_trues[i]) + '\n')
        if verbose:
            with open('checkpoints/logs.txt','a') as f:
                f.write('the start time for evaluation is {}'.format(time_now) + '\n')
                f.write('the avg_loss and auc is : {} and {}'.format(str(loss_avg),str(auc))+'\n')
        return y_pres_whole,y_trues
