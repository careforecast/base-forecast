import pandas as pd
import numpy as np
import random 
import pickle


def seq_data(file:str, save=False):
    """
    Arguments--
    file: .txt file that contains patientid and diagnosis codes
    save: if True it saves the list as pickle 
    
    Returns--
    patient_list: list contains dict with PatientsId as key and diagnosis codes as val
    seq_data: diagnosis codes
    """
    df = pd.read_csv(file,header=0,sep='\t')
    patient_id = df["PatientID"].unique()
    patient_list = []
    for patient in patient_id:
        df_temp = df.loc[df["PatientID"]==patient]
        code = list(df_temp["PrimaryDiagnosisCode"])
        temp_dict = {f"{patient}":code}
        patient_list.append(temp_dict)
    
    seq_data = []
    for x in patient_list:
        for _,val in x.items():
            seq_data.append(val)
    if save:
        with open(save,"wb") as writer:
            pickle.dump(seq_data,writer)
    return patient_list,seq_data

def get_indexed(seq_list):
    """
    Arguments--
    seq_list: list of sequence of codes
    
    Returns--
    seq_int: indexed data
    int2token: dict with index as key and token as value
    token2int: dict with 
    """
    seq = []
    for x in seq_list:
        seq.extend(x)
    seq = set(seq)
    print(f"total unique tokens: {len(seq)}")
    cnt = 0
    int2token = {}
    token2int = {}
    for x in seq:
        int2token[cnt] = x
        token2int[x] = cnt
        cnt += 1
        
    seq_int =[]
    for x in seq_list:
        seq_int.append([token2int[w] for w in x])
    
    return seq_int, (int2token,token2int)


def processing(file_path,data_size):
    """
    data preprocessing
    """
    print("Data preprocessing start...")
    patient_list, data = seq_data(file_path)
    data_int, (int2token,token2int) = get_indexed(data)

    with open(f'data/pickle/data-{data_size}.pickle', 'wb') as f:
        pickle.dump([data_int, int2token,token2int], f)
    print("Data preprocessing done...")
    


def padding(x_array, length, pad_value=0):
    """
    x_array: to be padded
    length: max length
    """
    len_ = len(x_array)
    len2pad = length-len_
    assert len2pad >= 0,"padding length should >= the array length"
    
    padded_x = np.pad(x_array,(0,len2pad),mode="constant",constant_values=pad_value)
    return padded_x


# Build dataset
class DiagnosisDataset(Dataset):
    def __init__(self,data_list, seq_length, pad_value=0 ,drop_len=1):
        self.seq_length = seq_length
        self.pad_value = pad_value
        self.drop_len = drop_len
        
        self.data_list = [x for x in data_list if len(x) > self.drop_len]
        self.input, self.target = self.input_target(self.data_list)
    
    def __getitem__(self,idx):
        inputs = np.array(self.input[idx]).astype(np.int64)
        inputs = padding(inputs,self.seq_length, pad_value=self.pad_value)
        
        targets = np.array(self.target[idx]).astype(np.int64)
        return inputs, targets

    def __len__(self):
        return len(self.target)
    
    def input_target(self,x):
        inputs = []
        targets = []
        for data_ in x:
            len_ = len(data_)
            inputs.append(data_[:(len_-1)])
            targets.append(data_[-1])
        return inputs, targets