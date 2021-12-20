import pandas as pd
import numpy as np
import os

#import from parent directory:
# import sys, os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)



def get_data(path,in_frame,load_data=False,aa_column_beta=1,v_beta_column=10,j_beta_column=16,frame_column=2,nc_column=0):   
    
    #file_name = 'data/seqs_inframe.txt'   
    file_name = 'data/whole_seqs.txt'   
    if load_data==False:
        beta_sequences = []
        v_beta,j_beta = [],[]
        file_lists = [path] if type(path) is str else path 
        print('loading Data')
        seps = ['\t' if f.endswith('.tsv') else ',' for f in file_lists]
        for i,file in enumerate(file_lists):
            print('here')
            f = pd.read_csv(file,sep=seps[i])
            col_v,col_j = f[f.columns[v_beta_column]].values,f[f.columns[j_beta_column]].values
            col_beta = f[f.columns[aa_column_beta]].values if in_frame else f[f.columns[nc_column]]
            frame_bool = f[f.columns[frame_column]].values 
            for k in range(len(col_beta)):
                if not PassFiltered_((col_beta[k],col_v[k],col_j[k]),frame_bool[k],in_frame):
                        continue  
                beta_sequences.append(col_beta[k])
                v_beta.append(col_v[k])
                j_beta.append(col_j[k])
                  
        with open(file_name,'w') as f_towrite:                
            for k in range(len(beta_sequences)):
                f_towrite.write(beta_sequences[k]+','+v_beta[k]+','+j_beta[k]+'\n')
    else :
        data = pd.read_csv(file_name,sep=',',names=['seq','v','j'])
        beta_sequences,v_beta,j_beta = data['seq'].values,data['v'].values,data['j'].values 
    return beta_sequences,v_beta,j_beta

def get_whole_data(files,aa_column_beta=1,v_beta_column=10,j_beta_column=16,frame_column=2,nc_column=0):
    beta_sequences = []
    v_beta,j_beta = [],[]
    print('loading Data')
    in_frame=True
    seps = ['\t' if f.endswith('.tsv') else ',' for f in files]
    unique_seqs = set()
    for i,file in enumerate(files):
        if i % 10 == 0:
            print('Have processed {} files'.format(i+1))
        f = pd.read_csv(file,sep=seps[i])
        col_v,col_j = f[f.columns[v_beta_column]].values,f[f.columns[j_beta_column]].values
        col_beta = f[f.columns[aa_column_beta]].values if in_frame else f[f.columns[nc_column]]
        frame_bool = f[f.columns[frame_column]].values 
        for k in range(len(col_beta)):
            if not PassFiltered_((col_beta[k],col_v[k],col_j[k]),frame_bool[k],in_frame):
                    continue  
            if col_beta[k] in unique_seqs:
                continue
            beta_sequences.append(col_beta[k])
            unique_seqs.add(col_beta[k])
            v_beta.append(col_v[k])
            j_beta.append(col_j[k])
    res = pd.DataFrame(columns=['seq','v','j'])
    res['seq'] = beta_sequences 
    res['v'] = v_beta
    res['j'] = j_beta
    res.to_csv('data/whole_seqs.tsv',sep='\t',index=False)

def get_whole_data_sonnia(files,aa_column_beta=1,v_beta_column=10,j_beta_column=16,frame_column=2,nc_column=0,max_length=30):
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
        
            if not PassFiltered_((col_beta[k],col_v[k],col_j[k]),frame_bool[k],in_frame,max_length):
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
    # res_unpro = pd.DataFrame(columns=['seq','v','j'])
    # res_unpro['seq'] = nn_unpro
    # res_unpro['v'] = v_un
    # res_unpro['j'] = j_un
    # res_unpro.to_csv('data/unpro.tsv',sep='\t',index=False)
    # res = pd.DataFrame(columns=['seq','v','j','nn'])
    # res['seq'] = beta_sequences 
    # res['v'] = v_beta
    # res['j'] = j_beta
    # res.to_csv('data/whole_seqs_unique_nn.tsv',sep='\t',index=False)
    beta_sequences,v_beta,j_beta,nn_beta = np.array(beta_sequences),np.array(v_beta),np.array(j_beta),np.array(nn_beta)
    infer_index = np.random.permutation(list(range(len(beta_sequences))))
    res_train = pd.DataFrame(columns=['seq','v','j','nn'])
    res_train['seq'] = beta_sequences[infer_index[:int(len(infer_index)/2)]]
    res_train['v'] = v_beta[infer_index[:int(len(infer_index)/2)]]
    res_train['j'] = j_beta[infer_index[:int(len(infer_index)/2)]]
    res_train['nn'] = nn_beta[infer_index[:int(len(infer_index)/2)]]
    res_train.to_csv('data/whole_seqs_train_nn_for_s.tsv',sep='\t',index=False)
    res_test = pd.DataFrame(columns=['seq','v','j','nn'])
    res_test['seq'] = beta_sequences[infer_index[int(len(infer_index)/2):]] 
    res_test['v'] = v_beta[infer_index[int(len(infer_index)/2):]] 
    res_test['j'] = j_beta[infer_index[int(len(infer_index)/2):]]
    res_test['nn'] = nn_beta[infer_index[int(len(infer_index)/2):]]
    res_test.to_csv('data/whole_seqs_test_nn_for_s.tsv',sep='\t',index=False)

def PassFiltered_(to_filter, frame_bool,in_frame,max_length):
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
    if j == 'TCRBJ02-05' :
        return False
    if seq =='CFFKQKTAYEQYF':
        return False
    return True

if __name__ == '__main__':
    get_whole_data_sonnia(['data_original/' + di for di in os.listdir('data_original')])
