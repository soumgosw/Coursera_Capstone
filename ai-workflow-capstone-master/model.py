import time,os,re,csv,sys,uuid,joblib
import pickle
from datetime import date
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from logger import update_predict_log, update_train_log

## specific variables for the versioning
if not os.path.exists(os.path.join(".","models")):
    os.mkdir("models") 

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Model Version 0.1"
SAVED_MODEL = os.path.join("models","model-{}.joblib".format(re.sub("\\.","_",str(MODEL_VERSION))))


def load_data():
    data_dir = os.path.join(".","data")
    df = pd.read_csv(os.path.join(data_dir,r"data.csv"))
       
    ## pull out the target and remove uneeded columns
    isSubscriber = df.pop('is_subscriber')
    b = np.zeros(isSubscriber.size)
    b[isSubscriber==0] = 1 
    df.drop(columns=['customer_id'],inplace=True)
    df.head()
    a = df

    return(a,b)

def get_preprocessor():
    """
    preprocessing pipeline
    """

    ## preprocessing pipeline
    numeric_features = ['age', 'num_streams']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])

    categorical_features = ['country', 'subscriber_type']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])


    return(preprocessor)

def train(test=False):
    """
    funtion to train the model
    """
   
    ## load data
    X,y = load_data()
    preprocessor = get_preprocessor()

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    ## Specify parameters and model
    param_grid = {
        'clf__n_estimators': [25,50,75,100],
        'clf__criterion':['gini','entropy'],
        'clf__max_depth':[2,4,6]
    }
   
    clf = ensemble.RandomForestClassifier()
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    print("... grid search")
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    params = grid.best_params_
    params = {re.sub("clf__","",key):value for key,value in params.items()}
    
    ## fit model on training data
    clf = ensemble.RandomForestClassifier(**params)
    pipe = Pipeline(steps=[('pre', preprocessor),
                           ('clf',clf)])
    
    pipe.fit(X_train,y_train)
    
    ## retrain using all data
    pipe.fit(X, y)

    if test:
        print("... saving test version of model")
        joblib.dump(pipe,os.path.join("models","test.joblib"))
    else:
        print("... saving model: {}".format(SAVED_MODEL))
        joblib.dump(pipe,SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models",'latest-train.pickle')
        with open(data_file,'wb') as tmp:
            pickle.dump({'y':y,'X':X},tmp)
    

def predict(query,model=None,test=False):
    """
    function to predict from model
    """
   
    ## validations
    if isinstance(query,dict):
        query = pd.DataFrame(query)
    elif isinstance(query,pd.DataFrame):
        pass
    else:
        raise Exception("ERROR: invalid input. {} was given".format(type(query)))

    ## features check
    features = sorted(query.columns.tolist())
    if features != ['age', 'country', 'num_streams', 'subscriber_type']:
        print("query features: {}".format(",".join(features)))
        raise Exception("ERROR: invalid features present") 
    
    ## load model if needed
    if not model:
        model = loadModel()
    
    ## check output
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    ## predict outcome
    y_pred = model.predict(query)
    y_proba = 'None'
           
    return({'y_pred':y_pred,'y_proba':y_proba})


def loadModel():
    """
    funtion to load model
    """

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)


if __name__ == "__main__":

    """
    unit test procedure
    """
    
    ## train the model
    train(test=True)

    ## load the model
    model = loadModel()
    
    ## example predict
    query = pd.DataFrame({'country': ['united_states','singapore','united_states'],
                          'age': [24,42,20],
                          'subscriber_type': ['aavail_basic','aavail_premium','aavail_basic'],
                          'num_streams': [8,17,14]
    })

    result = predict(query,model,test=True)
    y_pred = result['y_pred']
    print("predicted: {}".format(y_pred))
