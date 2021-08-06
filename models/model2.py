import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_model(train_df):
    #### feature map
    numeric_features = [
        'PatientPopulationPercentageBelowPoverty',
        'Age'
    ]
    categorical_features = ['PatientGender',
                            'PatientRace',
                            'PatientMaritalStatus',
                            'PatientLanguage',
                            'AdmissionID',
                            'PrimaryDiagnosisCode']
    vector_features= 'PrimaryDiagnosisDescription'

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    vector_transformer = Pipeline(steps=[
        ('vect', CountVectorizer( max_features=10, stop_words='english')),
        ('tfidf', TfidfTransformer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])#,
#             ('vect', vector_transformer,vector_features)])

    model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', RandomForestClassifier(max_depth=10,random_state=2))])
    
    #hyper parameter tune
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    rf_param = {'model__n_estimators': n_estimators,
                   'model__max_features': max_features,
                   'model__max_depth': max_depth,
                   'model__min_samples_split': min_samples_split,
                   'model__min_samples_leaf': min_samples_leaf,
                   'model__bootstrap': bootstrap}

    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    model = RandomizedSearchCV(model_rf, rf_param, n_iter=100, scoring='roc_auc', n_jobs=-1, cv=cv, random_state=1,verbose=1)
    
    X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Target'], axis=1), train_df.Target, test_size=0.2)

    model.fit(X_train,y_train)

    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)

    test_precision,test_recall=precision_score(y_test,y_pred_test),recall_score(y_test,y_pred_test)
    print(f"Precision: {test_precision},Recall: {test_recall}")
    train_precision,train_recall=precision_score(y_train,y_pred_train),recall_score(y_train,y_pred_train)
#     print(f"Precision: {train_precision},Recall: {train_recall}")
    
    # save the model to disk
    print("Model save..")
    filename = 'save_model/model.pkl'
    joblib.dump(model, filename)

    return model,(test_precision,test_recall),(train_precision,train_recall)
