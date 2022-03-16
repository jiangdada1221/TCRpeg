import pandas as pd
import numpy as np
import os

'''
How to use this code for getting the universal TCR pool from Emerson et al.:
First put the 743 files of individual repertoires under the folder 'data_original/', and then
in terminal, type:
python process_data.py

After execution, it will generate two files 'whole_seqs_train_nn.tsv' and 'whole_seqs_train_nn.tsv' under the 'data/' folder (So you need to create folder 'data' in advance)
'''


vs_default = ['TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2','TRBV11-3','TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 
        'TRBV16', 'TRBV18','TRBV19','TRBV2', 'TRBV20-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV3-1', 'TRBV30',
'TRBV4-1', 'TRBV4-2','TRBV4-3',
 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8','TRBV6-1', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6','TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3',
  'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
js_default = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3','TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6','TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-6', 'TRBJ2-7']

def get_whole_data(files,aa_column_beta=1,v_beta_column=10,j_beta_column=16,frame_column=2,nc_column=0,max_length=30,filter_genes=True):
    #Pool the universal TCR pool from Emerson et al.

    #filter_genes: whether to filter genes that not in vs_default and js_default. This step is just used to follow soNNia and TCRvae since 
    #they all excludes genes that not in these two lists

    beta_sequences = []
    v_beta,j_beta = [],[]
    nn_beta = []
    nn_unpro,v_un,j_un = [],[],[]
    print('loading Data')
    in_frame=True
    seps = ['\t' if f.endswith('.tsv') else ',' for f in files]
    unique_seqs = set() # to store nn now
    count = 0

    for i,file in enumerate(files):
        if i % 10 == 0:
            print('Have processed {} files'.format(i+1))
        f = pd.read_csv(file,sep=seps[i])
        col_nn = f[f.columns[nc_column]].values
        col_v,col_j = f[f.columns[v_beta_column]].values,f[f.columns[j_beta_column]].values
        col_beta = f[f.columns[aa_column_beta]].values if in_frame else f[f.columns[nc_column]]
        frame_bool = f[f.columns[frame_column]].values 
        frames_out = [f for f in frame_bool if f != 'In' or f!='in']
        # if len(frames_out) <= 10000:
        #     count += 1
        #     continue
        for k in range(len(col_beta)):
            if frame_bool[k] != 'In' or frame_bool[k] != 'in':
                if col_v[k] !='unresolved' and col_j[k] != 'unresolved':
                    if type(col_v[k]) is str and type(col_j[k]) is str:
                        if col_j[k] != 'TCRBJ02-05':
                            #nn_unpro.append(col_nn[k])
                            # v_un.append(col_v[k])
                            # j_un.append(col_j[k])
                            pass
        
            if not PassFiltered_((col_beta[k],col_v[k],col_j[k]),frame_bool[k],in_frame,max_length,filter_genes):
                    continue  
            if col_nn[k] in unique_seqs:
                continue
            beta_sequences.append(col_beta[k])
            unique_seqs.add(col_nn[k])
            v_beta.append(col_v[k])
            j_beta.append(col_j[k])
            nn_beta.append(col_nn[k])
        print(len(beta_sequences))
    print(count)
    beta_sequences,v_beta,j_beta,nn_beta = np.array(beta_sequences),np.array(v_beta),np.array(j_beta),np.array(nn_beta)
    infer_index = np.random.permutation(list(range(len(beta_sequences))))
    res_train = pd.DataFrame(columns=['seq','v','j','nn'])
    res_train['seq'] = beta_sequences[infer_index[:int(len(infer_index)/2)]]
    res_train['v'] = v_beta[infer_index[:int(len(infer_index)/2)]]
    res_train['j'] = j_beta[infer_index[:int(len(infer_index)/2)]]
    res_train['nn'] = nn_beta[infer_index[:int(len(infer_index)/2)]]
    res_train.to_csv('data/whole_seqs_train_nn.tsv',sep='\t',index=False,compression='gzip')
    res_test = pd.DataFrame(columns=['seq','v','j','nn'])
    res_test['seq'] = beta_sequences[infer_index[int(len(infer_index)/2):]] 
    res_test['v'] = v_beta[infer_index[int(len(infer_index)/2):]] 
    res_test['j'] = j_beta[infer_index[int(len(infer_index)/2):]]
    res_test['nn'] = nn_beta[infer_index[int(len(infer_index)/2):]]
    res_test.to_csv('data/whole_seqs_test_nn.tsv',sep='\t',index=False,compression='gzip')

def PassFiltered_(to_filter, frame_bool,in_frame,max_length,filter_genes):
    ### to check whether the row of data is valid
    seq,v,j = to_filter
    if in_frame:
        if frame_bool != 'In':
            return False             
    # else :
    #     if frame_bool != 'Out':
    #         return False
    if type(seq) is not str:            
        return False
    if type(v) is not str or type(j) is not str:
        return False
    if len(seq) > max_length or '*' in seq or 'C' != seq[0] or ('F' != seq[-1] and 'YV' != seq[-2:]):
        return False
    if v =='unresolved' or j == 'unresolved':
        return False

    if j == 'TCRBJ02-05' : #based on data preprocessing step in soNNia
        return False
    if seq =='CFFKQKTAYEQYF': #based on data preprocessing step in soNNia
        return False
    if filter_genes:
        if (v not in vs_default) or (j not in js_default):
            return False
    return True

if __name__ == '__main__':
    get_whole_data(['data_original/' + di for di in os.listdir('data_original')])
