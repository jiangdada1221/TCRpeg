from tcrpeg.model import FC_NN_large , FC_NN_medium, FC_NN_small
import torch              
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import average_precision_score as AUPRC
import copy

class classification:
    '''
    TCRpeg-c: Simple FCN models that utilize the embedding from trained tcrpeg model
    '''
    def __init__(self,tcrpeg,embedding_size,device='cuda:0',dropout=0.0,model_size='large',last_layer=False,batch_norm=True):
        '''
        @tcrpeg: the pretrained model
        @embedding_size: the size of the embedding vectors
        @device: specify using GPU or CPU
        @dropout: dropout rate for FCN models
        @model_size: model size of the FCN model, options: {'small','medium','large'}
        @last_layer: whether using only the last hidden layer or not. If set to False, embedding_size should be #layers*hidden_size 
        '''
        self.emb_model = tcrpeg
        self.emb_model.model.eval()
        self.device = device        
        if model_size=='small':
            self.model = FC_NN_small(embedding_size,last_layer,device)
        elif model_size == 'medium':
            self.model = FC_NN_medium(embedding_size,dropout,last_layer,device,batch_norm)
        else :
            assert model_size == 'large'
            self.model = FC_NN_large(embedding_size,dropout,last_layer,device,batch_norm)
        self.model = self.model.to(device)

        self.criterion = nn.BCELoss()
        self.last_layer = last_layer
        self.device = device
        
    @classmethod
    def valid_seq(self,seq):
        '''
        To test whether a given seq is a valid TCR_beta sequence
        '''
        if type(seq) is not str:
            return False
        if len(seq) <= 2:

            return False
        if seq[-1] != 'F' and seq[-2:] != 'YV':

            return False
        if seq[0] != 'C':

            return False
            
        if ('s' in seq) or ('e' in seq):
            return False
        return True

    def load_model(self,path):
        #load the pre_trained model
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save(self,path):
        ###save the classification model to path
        torch.save(self.model.state_dict(), path)
    
    def train(self,x_train,y_train,epochs,batch_size,lr,val_split=0.1,save_name=None,metric='auroc'):
        '''
        @x_train: a list of seqs
        @y_train: a list containing the labels (0s and 1s)
        @epochs: number of epoch
        @batch_size: batch_size
        @lr: initial learning rate
        @val_split: split ratio for constructing the validation set
        @save_name: the name to save the best model (will be stored as save_name.pth)
        '''
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        infer = list(range(len(x_train)))
        length = len(x_train)
        np.random.shuffle(infer)
        to_split = True if val_split != 0 else False
        if not isinstance(x_train,np.ndarray):
            x_train,y_train = np.array(x_train),np.array(y_train)
        if to_split:
            x_val,y_val = x_train[infer[:int(length* val_split)]],y_train[infer[:int(length* val_split)]]
            x_train,y_train = x_train[infer[int(length* val_split):]],y_train[infer[int(length* val_split):]]
        else :
            x_train,y_train = x_train[infer],y_train[infer]

        best_metric = -1
        best_model = None

        infer = list(range(len(x_train))) #since the size of x_train has changed
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
                x_batch = self.emb_model.get_embedding(x_batch,last_layer  = self.last_layer) #B x embedding_size
                x_batch = torch.tensor(x_batch).to(self.device)

                y_pres = self.model(x_batch).view(batch_size)
                loss = self.criterion(y_pres,y_batch)
                optimizer.zero_grad()
                loss_.append(loss.item()*batch_size)
                loss.backward()
                optimizer.step()
                y_pres_whole = y_pres_whole + list(y_pres.detach().cpu().numpy())
            # auc = AUC(x_test,)
            print('mean loss :',sum(loss_)/len(x_train))
            if to_split:
                auc,aup,__,_ = self.evaluate(x_val,y_val,batch_size=batch_size)
                if metric == 'auroc':
                    metric_tmp = auc
                else :
                    metric_tmp = aup
                if metric_tmp > best_metric:
                    best_metric = metric_tmp
                    best_model = copy.deepcopy(self.model)
            if epoch % (epochs//2) == 0 and epoch != 0:
                print('The learning rate is reduced')
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
        print('end training')
        if to_split:
            self.model = best_model.to(self.device)
        else :
            best_model = self.model
        self.model.eval()
        
        if save_name is not None:
            torch.save(best_model.state_dict(),'results/'+save_name + '.pth')

    def train_tcrgp(self,x_train,y_train,emb_x,epochs,batch_size,lr,val_split=0.1,save_name=None,metric='auroc'):
        '''
        @x_train: a list of seqs
        @y_train: a list containing the labels (0s and 1s)
        @epochs: number of epoch
        @batch_size: batch_size
        @lr: initial learning rate
        @val_split: split ratio for constructing the validation set
        @save_name: the name to save the best model (will be stored as save_name.pth)
        '''
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        infer = list(range(len(x_train)))
        length = len(x_train)
        np.random.shuffle(infer)
        to_split = True if val_split != 0 else False
        if not isinstance(x_train,np.ndarray):
            x_train,y_train = np.array(x_train),np.array(y_train)
        if to_split:
            x_val,y_val = x_train[infer[:int(length* val_split)]],y_train[infer[:int(length* val_split)]]
            x_train,y_train = x_train[infer[int(length* val_split):]],y_train[infer[int(length* val_split):]]
            x_train_emb,x_val_emb = emb_x[infer[int(length* val_split):]],emb_x[infer[:int(length* val_split)]]
        else :
            x_train,y_train = x_train[infer],y_train[infer]
            x_train_emb = emb_x[infer]

        best_metric = -1
        best_model = None

        infer = list(range(len(x_train))) #since the size of x_train has changed
        for epoch in range(epochs):
            self.model.train()
            print('begin epoch :',epoch+1)
            np.random.shuffle(infer)
            x_train,y_train = x_train[infer],y_train[infer]
            x_train_emb = x_train_emb[infer]
            loss_ = []
            y_pres_whole = []
            for iter in tqdm(range(len(infer)//batch_size)):
                x_batch = x_train[iter*batch_size : (iter+1) * batch_size]
                y_batch = y_train[iter*batch_size : (iter+1) * batch_size]
                x_batch = x_train_emb[iter*batch_size : (iter+1) * batch_size]
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                # x_batch = self.emb_model.get_embedding(x_batch,last_layer  = self.last_layer) #B x embedding_size
                x_batch = torch.tensor(x_batch).to(self.device)

                y_pres = self.model(x_batch).view(batch_size)
                loss = self.criterion(y_pres,y_batch)
                optimizer.zero_grad()
                loss_.append(loss.item()*batch_size)
                loss.backward()
                optimizer.step()
                y_pres_whole = y_pres_whole + list(y_pres.detach().cpu().numpy())
            # auc = AUC(x_test,)
            print('mean loss :',sum(loss_)/len(x_train))
            if to_split:
                auc,aup,_,_ = self.evaluate(x_val,y_val,batch_size=batch_size)
                if metric == 'auroc':
                    metric_tmp = auc
                else :
                    metric_tmp = aup
                if metric_tmp > best_metric:
                    best_metric = metric_tmp
                    best_model = copy.deepcopy(self.model)
            if epoch % (epochs//2) == 0 and epoch != 0:
                print('The learning rate is reduced')
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2
        print('end training')
        if to_split:
            self.model = best_model.to(self.device)
        else :
            best_model = self.model
        self.model.eval()
        
        if save_name is not None:
            torch.save(best_model.state_dict(),'results/'+save_name + '.pth')

    def evaluate(self,x_test,y_test,batch_size,record_path=None):
        '''
        Evaluation on test set
        @x_train: a list of seqs
        @y_train: a list containing the labels (0s and 1s)
        @batch_size: batch_size
        @record_path: write the values of evaluation metrics to the file under given path specified by record_path

        @return: (AUC,AUPRC, a list containing the predicted scores, a list containing the corresponding true labels)
        '''
        time_now = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
        y_pres_whole = []
        y_trues = []
        self.model.eval()
        with torch.no_grad():
            loss_ = []
            for iter in tqdm(range(int(len(x_test)/batch_size))):
                end = (iter+1) * batch_size if iter != int(len(x_test)/batch_size)-1 else len(x_test) #to make sure every data point is used
                x_batch = x_test[iter*batch_size : end]
                y_batch = y_test[iter*batch_size : end]
                y_trues += list(y_batch)
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                batch_size_temp = len(x_batch)
                x_batch = self.emb_model.get_embedding(x_batch,last_layer  = self.last_layer)
                x_batch = torch.tensor(x_batch).to(self.device)

                y_pres = self.model(x_batch).view(batch_size_temp)
                loss_.append(self.criterion(y_pres,y_batch).item()*batch_size_temp)
                y_pres = y_pres.detach().cpu().numpy()
                y_pres_whole = y_pres_whole + list(y_pres)
            loss_avg = sum(loss_) / len(x_test)
            auc = round(AUC(y_trues,y_pres_whole),3)
            aup = round(AUPRC(y_trues,y_pres_whole),3)
            #auc = self.acc(y_trues,y_pres_whole)
            print('evaluation, avg_loss and auc is :{} and {}'.format(str(loss_avg),str(auc)))

        if record_path is not None:
            assert record_path.endswith('.txt'), 'the record file should be in txt format'
            with open(record_path,'a') as f:
                #f.write('the start time for evaluation is {}'.format(time_now) + '\n')
                f.write('the avg_loss, auroc, and auprc are : {}, {}, and {}'.format(str(loss_avg),str(auc), str(aup))+'\n')
        return auc,aup,y_pres_whole,y_trues

    def evaluate_tcrgp(self,x_test,y_test,emb_x,batch_size,record_path=None):
        '''
        Evaluation on test set
        @x_train: a list of seqs
        @y_train: a list containing the labels (0s and 1s)
        @batch_size: batch_size
        @record_path: write the values of evaluation metrics to the file under given path specified by record_path

        @return: (AUC,AUPRC, a list containing the predicted scores, a list containing the corresponding true labels)
        '''
        time_now = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
        y_pres_whole = []
        y_trues = []
        self.model.eval()
        with torch.no_grad():
            loss_ = []
            for iter in tqdm(range(int(len(x_test)/batch_size))):
                end = (iter+1) * batch_size if iter != int(len(x_test)/batch_size)-1 else len(x_test) #to make sure every data point is used
                x_batch = x_test[iter*batch_size : end]
                y_batch = y_test[iter*batch_size : end]
                y_trues += list(y_batch)
                x_batch = emb_x[iter*batch_size : end]
                y_batch = torch.FloatTensor(y_batch).to(self.device)
                batch_size_temp = len(x_batch)
                #x_batch = self.emb_model.get_embedding(x_batch,last_layer  = self.last_layer)
                x_batch = torch.tensor(x_batch).to(self.device)

                y_pres = self.model(x_batch).view(batch_size_temp)
                loss_.append(self.criterion(y_pres,y_batch).item()*batch_size_temp)
                y_pres = y_pres.detach().cpu().numpy()
                y_pres_whole = y_pres_whole + list(y_pres)
            loss_avg = sum(loss_) / len(x_test)
            auc = round(AUC(y_trues,y_pres_whole),3)
            aup = round(AUPRC(y_trues,y_pres_whole),3)
            #auc = self.acc(y_trues,y_pres_whole)
            print('evaluation, avg_loss and auc is :{} and {}'.format(str(loss_avg),str(auc)))

        if record_path is not None:
            assert record_path.endswith('.txt'), 'the record file should be in txt format'
            with open(record_path,'a') as f:
                #f.write('the start time for evaluation is {}'.format(time_now) + '\n')
                f.write('the avg_loss, auroc, and auprc are : {}, {}, and {}'.format(str(loss_avg),str(auc), str(aup))+'\n')
        return auc,aup,y_pres_whole,y_trues
