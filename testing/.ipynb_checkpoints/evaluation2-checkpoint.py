import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score

def evaluation_data(df_test,model_path="../save_model/model.pkl", save=False):
    loaded_model = joblib.load(model_path)
    df_test['Target'].replace({False: 0, True: 1}, inplace=True)
    for col in df_test.columns:
        df_test[col]=df_test[col].astype(str)
        df_test[col]=df_test[col].map(lambda x: x.lower())
        
    y_pred = loaded_model.predict(df_test.drop(['Target'],axis=1)) 
    
    df_test["Target"]=df_test["Target"].astype(int)
    test_precision= precision_score(df_test.Target,y_pred)
    test_recall = recall_score(df_test.Target,y_pred)  
    df_test["y_pred"] = y_pred
    print(f"Precision: {test_precision},Recall: {test_recall}")
    if save:
        df_test.to_csv("evaluation.csv")
    return df_test

def infer(test_df,model_path="save_model/model.pkl"):
    loaded_model = joblib.load(model_path)
    y_pred=loaded_model.predict(test_df)
    define={
        0:"No Need To Admit",
        1:"Please Admit"
    }
    return [define[x] for x in y_pred]
                                    
                                    
                                      
                                    
                            
        

