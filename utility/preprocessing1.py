from itertools import combinations
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
    print(f"length={len(data_int)}")
    
def load_pickle(data_size):
    try:
       
        with open(f'data/pickle/data-{data_size}.pickle', 'rb') as f:
            data_int,int2token,token2int= pickle.load(f)
        print("Load data from pickle")
        print(f"length={len(data_int)}")
        return data_int,int2token,token2int
     
    except:
        print("issue in pikle file loading")

# data augmentaion
def augmentaion(list_):
    """
    list_: list of items
    """
    data = []
    i = 0
    while i < len(list_):
        for l in range(2,len(list_)+1):
            for subset in combinations(list_[i:],l):
                data.append(list(subset))
                break
        i += 1
    return data

def get_augmentaion(data_int):
    data_aug = []
    for x in data_int:
        x = augmentaion(x)
        data_aug.extend(x)
    print(f"Total={len(data_aug)}")
    return data_aug

def train_test_split(data,ratio=0.20,random_seed=None):
    """
    Arguments--
    data: list of sequence
    ration: train-test ratio. Default is 0.20
    Returns:
    train: tarining data
    test: test data
    """
    if random_seed:
        random.seed(random_seed)
    
    k = int(ratio*len(data))
    test = random.choices(data,k=k)
    train = [x for x in data if x not in test]
    
    return train,test

