from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def evaluation_data(X_test,y_test,size, save=False):
    model_path="save_model/model_{}/model.h5".format(size)
    loaded_model = load_model(model_path)
    y_pred = loaded_model.predict(X_test) 
    print('Evaluation result: ',loaded_model.evaluate(X_test,y_test))
    y_pred=list(y_pred.flatten())
    y_test=list(y_test.flatten())
    output = pd.DataFrame({"Actual": y_test, "Predictions": y_pred})
    output['Predictions']=output['Predictions'].apply(lambda x: 0 if x<0.5 else 1)
    output['Actual']=output['Actual'].apply(lambda x: int(x))
    if save:
        output.to_csv("output.csv")
    return output

def infer(test_df,size):
    test = np.expand_dims(test_df.values,axis = 2)
    model_path="save_model/model_{}/model.h5".format(size)
    loaded_model = load_model(model_path)    
    y_pred=loaded_model.predict(test)
    y_pred=list(y_pred.flatten())
    test_df['Predictions']=y_pred
    test_df['Predictions']=test_df['Predictions'].apply(lambda x: 0 if x<0.5 else 1)
    preds=test_df['Predictions'].values.tolist()
    return preds
                                    
if __name__ == '__main__':
    DATA_SIZE=10000
    BASE_PATH=f'data/{DATA_SIZE}'
    train_df=pd.read_csv(f'{BASE_PATH}/train.csv')
    #X_train,y_train,X_test,y_test,X_val,y_val=split_data(train_df)
    df_pred=evaluation_data(X_test,y_test,DATA_SIZE,False)
    print(df_pred)

