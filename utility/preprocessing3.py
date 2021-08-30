## 30 days classification
import os
import glob
from tqdm import tqdm
import pandas as pd
from imblearn.over_sampling import SMOTENC 
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

import sys

def clean_data(path):
    print("process start..")
    df_admi_table=pd.read_csv(f'{path}/AdmissionsCorePopulatedTable.txt', 
                              names=['PatientID','AdmissionID','AdmissionStartDate','AdmissionEndDate'], skiprows=1, sep='\t')
    df_admi_dia=pd.read_csv(f'{path}/AdmissionsDiagnosesCorePopulatedTable.txt', 
                            names=['PatientID','AdmissionID','PrimaryDiagnosisCode','PrimaryDiagnosisDescription'], skiprows=1, sep='\t')
    df_pop_table=pd.read_csv(f'{path}/PatientCorePopulatedTable.txt', 
                             names=['PatientID','PatientGender','PatientDateOfBirth','PatientRace','PatientMaritalStatus','PatientLanguage','PatientPopulationPercentageBelowPoverty'], skiprows=1, sep='\t')
    path_csv=f"{path}/process/csv_dir"
    os.makedirs(path_csv, exist_ok = True) 
    pid=pd.unique(df_admi_table['PatientID']).tolist()
    raw=pd.DataFrame({"PatientID":[],'AdmissionID':[],'AdmissionStartDate':[],'AdmissionEndDate':[],'new':[]})
    for i in tqdm(pid):
        _df=df_admi_table[df_admi_table['PatientID']==i]
        _df=_df.sort_values(by = ['AdmissionID'])
        if len(_df)<=1:
            #print(_df)
            _df['target_days']=[99999999999999]
            _df['AdmissionStartDate']= pd.to_datetime(_df['AdmissionStartDate'])
            _df['AdmissionEndDate']= pd.to_datetime(_df['AdmissionEndDate'])

        else:
            p=_df['AdmissionEndDate'].tolist()
            p.insert(0, datetime.strptime('Jan 1 1900 1:33PM', '%b %d %Y %I:%M%p'))
            del p[-1]
            _df['new']=p
            df=_df.copy()
            _df['AdmissionStartDate']= pd.to_datetime(_df['AdmissionStartDate'])
            _df['AdmissionEndDate']= pd.to_datetime(_df['AdmissionEndDate'])
            _df['new']= pd.to_datetime(_df['new'])
            #date
            difference=(_df['AdmissionStartDate'] - _df['new'])
            days=[]
            for x in difference:
                days.append(int(str(x).split(' ')[0]))
            _df['target_days']=days
            _df.drop('new', axis=1, inplace=True)
        
        _df['days']=(_df['AdmissionEndDate'] - _df['AdmissionStartDate'])
        _df['days']=_df['days'].apply(lambda x: int(str(x).split(' ')[0]))
        _df.drop('AdmissionEndDate', axis=1, inplace=True)
        _df.to_csv(f"{path_csv}/{i}.csv",index=False)
            
    ar=[]
    file=glob.glob(f"{path_csv}/*.csv")
    for f in file:
        ar.append(pd.read_csv(f))
    df=pd.concat(ar)
    
    df_=pd.merge(df, df_admi_dia, on=["PatientID", "AdmissionID"])
    fps=pd.merge(df_pop_table, df_, on=["PatientID"])
    fps.to_csv(f"{path}/process/final_{path.split('/')[-1]}.csv",index=False)
    
    print('preprocessing done')
    
def preprocessing(df,days,feature):
    #df["target_days"]=df["target_days"].apply(lambda x: int(x))
    df['Target']=df['target_days']<=days
    df['Target'].replace({False: 0, True: 1}, inplace=True)
    #age
    df['PatientDateOfBirth']= pd.to_datetime(df['PatientDateOfBirth'])
    df['AdmissionStartDate']= pd.to_datetime(df['AdmissionStartDate'])
    from dateutil.relativedelta import relativedelta
    df['Age'] = [relativedelta(a, b).years for a, b in zip( df['AdmissionStartDate'],df['PatientDateOfBirth'])]
    df.drop(['AdmissionStartDate','PatientDateOfBirth','target_days'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = shuffle(df)
    return df[feature]


def smote(train_df):
    oversample = SMOTENC(categorical_features=[0,1,2,4,6,8],sampling_strategy='minority',k_neighbors=3)
    y=train_df[['Target']]
    X=train_df.drop('Target', axis=1, inplace=False)

    X, y = oversample.fit_resample(X, y)
    X['days']=X['days'].astype(int)
    df = X.assign(Target=y)
    return df

def normalise_data(df,BASE_PATH):
    pid=pd.unique(df['PatientID']).tolist()
    for i in pid:
        temp=df[df['PatientID']==i]
        if len(temp['AdmissionID'].tolist()) == len(set(temp['AdmissionID'].tolist())):
            continue
        val_counts=temp['AdmissionID'].value_counts(ascending=True)
        ad_ids=val_counts.index.tolist()
        for idd in ad_ids:
            if val_counts[idd] == 1:
                gender=temp[temp['AdmissionID']==idd]['PatientGender'].values[0]
                race=temp[temp['AdmissionID']==idd]['PatientRace'].values[0]
                status=temp[temp['AdmissionID']==idd]['PatientMaritalStatus'].values[0]
                df.loc[df.PatientID == i, 'PatientGender'] = gender
                df.loc[df.PatientID == i, 'PatientRace'] = race
                df.loc[df.PatientID == i, 'PatientMaritalStatus'] = status
                break
            
    
    return preprocess_column(df,BASE_PATH,True)

def preprocess_column(df,BASE_PATH,save=False):
    df.replace({
                'PatientGender': {
                    'Male': 1,
                    'Female': 0},
                'PatientRace': {
                    'African American': 0,
                    'Asian':1,
                    'White':2,
                    'Unknown':3
                },
                'PatientMaritalStatus': {
                    'Married': 0,
                    'Single':1,
                    'Separated':2,
                    'Divorced':3,
                    'Widowed':4,
                    'Unknown':5
                }},inplace=True)
    if save:
        ids=df['PatientID'].values.tolist()
        diag=df['PrimaryDiagnosisCode'].values.tolist()
        df['PatientID'] = df['PatientID'].astype('category').cat.codes
        df['PrimaryDiagnosisCode'] = df['PrimaryDiagnosisCode'].astype('category').cat.codes
        codes=df['PatientID'].values.tolist()
        diag_code=df['PrimaryDiagnosisCode'].values.tolist()
        mapping = pd.DataFrame({"PatientID": ids, "Patient_Code": codes,"PrimaryDiagnosisCode": diag, "Diag_Code": diag_code})
        mapping.to_csv(f'{BASE_PATH}/mappings.csv',index=False)
    return df
            
def split_data(df):
    #np.random.seed(100)
    seed=20
    train_ratio = 0.80
    validation_ratio = 0.195
    test_ratio = 0.005
    X = np.expand_dims(df.values[:,:-1],axis = 2)
    y = df.values[:,-1:]
    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio,random_state=seed)
    
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=seed) 
    return (x_train,y_train,x_test,y_test,x_val,y_val)
        
if __name__ == '__main__':
    DATA_SIZE=10000
    BASE_PATH=f'data/{DATA_SIZE}'
    #clean_data(BASE_PATH)
    train_df=pd.read_csv(f'{BASE_PATH}/train.csv')
    #normalise_data(train_df)
    #a(train_df)
    #print(train_df[train_df['PatientID'] == 'cd2adb1b-97f7-4ef6-bc5c-3e0ec562a06f'])
    split_data(train_df)