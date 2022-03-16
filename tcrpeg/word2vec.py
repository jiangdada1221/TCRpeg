import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def word2vec(path,epochs,batch_size,device,record_path=None,lr=0.0001,window_size=2,embedding_dims = 32):
    '''
    Training the word2vec for embedding of amino acids
    @path: path to the file that records the sequences
    @batch_size: batch_size    
    @device: GPU or CPU; 'cuda:0'-GPU, 'cpu'-CPU
    @record_path: path to record the embedding for each AA; should ended with '.txt'
    @lr: initial learning rate
    @window_size: the window_size used for constructing training pairs (see word2vec paper for details)

    #return: the embedding (a numpy array)
    '''
    def get_input_layer(word_idxs):
        #x = torch.zeros(vocabulary_size).float()
        x = torch.zeros((len(word_idxs),vocabulary_size)).float()
        x[list(range(len(word_idxs))),word_idxs] = 1.0  # n x emb
        assert torch.sum(x) == len(word_idxs)
        return x #one hot vector
    #assert record_path.endswith('.txt')

    #list of a string is served as default tokenizer
    aas = 'sACDEFGHIKLMNPQRSTVWYe'
    aa2idx = {aas[i]:i for i in range(len(aas))}
    idx2aa = {v: k for k, v in aa2idx.items()}
    vocabulary_size = len(aas)
    batch_size = batch_size
    device = device    
    if isinstance(path,list) or isinstance(path,np.ndarray):
        tcr_sequences = np.array(path)
    else :
        tcr_sequences = pd.read_csv(path,sep='\t')['seq'].values

    tokenized_seqs = [['s']+list(seq)+['e'] for seq in tcr_sequences]

    idx_pairs = []
    # for each sentence
    for seq in tokenized_seqs:
        indices = [aa2idx[word] for word in seq]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs) # N x 2
    W1 = Variable(torch.randn(embedding_dims, vocabulary_size,device=device).float(), requires_grad=True) 
    W2 = Variable(torch.randn(vocabulary_size, embedding_dims,device=device).float(), requires_grad=True) 
    num_epochs = epochs
    learning_rate = lr

    for epo in range(num_epochs):
        loss_val = 0        
        iter = len(idx_pairs) // batch_size 
        for iter in tqdm(range(iter)):
            #x = Variable(get_input_layer(data)).float()
            x = Variable(get_input_layer(idx_pairs[iter*batch_size:(iter+1)*batch_size,0])).float().to(device)  #nxv
        
            y_true = Variable(torch.from_numpy(np.array(idx_pairs[iter*batch_size:(iter+1)*batch_size,1])).long()).to(device)
            
            z1 = torch.matmul(W1, torch.transpose(x,0,1))
            z2 = torch.matmul(W2, z1) # |v| x batch_size
            
            log_softmax = F.log_softmax(z2, dim=0).permute([1,0]) #batch_size x |v|
            loss = F.nll_loss(log_softmax, y_true)
            loss_val += loss.item()
            loss.backward()
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()
            
        if epo % 2 == 0:    
            print(f'Loss at epo {epo+1}: {loss_val * 32 /len(idx_pairs)}')
        if epo % 40 == 0:
            learning_rate *= 0.2

    embedding = W1.cpu().data.numpy()
    if record_path is not None:
        with open(record_path, 'w') as f:
            for i in range(22):
                f.write(aas[i]+',')
                for j in range(embedding_dims):
                    f.write(str(embedding[j,i]))
                    if j != embedding_dims-1:
                        f.write(',')
                f.write('\n')
    return embedding


