## 30 days classification
import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.utils import resample
from sklearn.utils import shuffle


def clean_data(path):
    print("process start..")
    df_admi_table=pd.read_csv(f'{path}/AdmissionsCorePopulatedTable.txt', 
                              names=['PatientID','AdmissionID','AdmissionStartDate','AdmissionEndDate'], skiprows=1, sep='\t')
    df_admi_dia=pd.read_csv(f'{path}/AdmissionsDiagnosesCorePopulatedTable.txt', 
                            names=['PatientID','AdmissionID','PrimaryDiagnosisCode','PrimaryDiagnosisDescription'], skiprows=1, sep='\t')
    df_pop_table=pd.read_csv(f'{path}/PatientCorePopulatedTable.txt', 
                             names=['PatientID','PatientGender','PatientDateOfBirth','PatientRace','PatientMaritalStatus','PatientLanguage','PatientPopulationPercentageBelowPoverty'], skiprows=1, sep='\t')

    path_csv="data/process/csv_dir"
    os.makedirs(path_csv, exist_ok = True) 
    pid=pd.unique(df_admi_table['PatientID']).tolist()
    raw=pd.DataFrame({"PatientID":[],'AdmissionID':[],'AdmissionStartDate':[],'AdmissionEndDate':[],'new':[]})
    
    for i in tqdm(pid):
        _df=df_admi_table[df_admi_table['PatientID']==i]
        _df=_df.sort_values(by = ['AdmissionID']) 
        if len(_df)<=1:
            pass
        else:
            p=_df['AdmissionStartDate'].tolist()
            p.append(p.pop(0))
            _df['new']=p
            _df[:-1].to_csv(f"{path_csv}/{i}.csv",index=False)

    ar=[]
    file=glob.glob(f"{path_csv}/*.csv")
    for f in file:
        ar.append(pd.read_csv(f))
    df=pd.concat(ar)
    
    df['AdmissionStartDate']= pd.to_datetime(df['AdmissionStartDate'])
    df['AdmissionEndDate']= pd.to_datetime(df['AdmissionEndDate'])
    df['new']= pd.to_datetime(df['new'])
    #date
    difference=(df['new'] - df['AdmissionEndDate'])
    days=[]
    for x in difference:
        days.append(int(str(x).split(' ')[0]))
    df['days']=days

    df_=pd.merge(df, df_admi_dia, on=["PatientID", "AdmissionID"])
    fps=pd.merge(df_pop_table, df_, on=["PatientID"])
    fps.to_csv(f"data/process/final_{path.split('/')[-1]}.csv",index=False)
    
    print('preprocessing done')
    

def lower_case(df):
    for col in df.columns:
        df[col]=df[col].astype(str)
        df[col]=df[col].map(lambda x: x.lower())
    return df
        
def preprocessing(df,days,feature):
    df["days"]=df["days"].astype(int)
    df['Target']=df['days']<=days
    df['Target'].replace({False: 0, True: 1}, inplace=True)
    #age
    df['PatientDateOfBirth']= pd.to_datetime(df['PatientDateOfBirth'])
    df['AdmissionEndDate']= pd.to_datetime(df['AdmissionEndDate'])
    from dateutil.relativedelta import relativedelta
    df['Age'] = [relativedelta(a, b).years for a, b in zip( df['AdmissionEndDate'],df['PatientDateOfBirth'])]
    df.dropna(inplace=True)
    df = shuffle(df)
    return df[feature]


def upsample(train_df,size=1000):
    
    # Separate majority and minority classes
    df_majority = train_df[train_df.Target==False]
    df_minority = train_df[train_df.Target==True]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=size,    # to match majority class
                                     random_state=123) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    train_false = shuffle(train_df[train_df['Target']==False])
    
    final_df=pd.concat([df_upsampled[df_upsampled['Target']==True],train_false[:size]])
    
    train = shuffle(final_df)
    
    return train
    