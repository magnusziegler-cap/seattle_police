from fastapi import FastAPI
import uvicorn
import joblib
import json

import pandas as pd
import numpy as np

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

## Statics
PATH_TO_MODEL = 'models\\model_and_encoder.joblib'
PATH_TO_DATA = 'data\\police_stops.pickle'

## start app
app = FastAPI(title="Seattle Police What If Scenario Generator/Interpreter")

## functions
@app.get("/")
def home():
    """Server root

    Returns:
        _type_: _description_
    """
    return {"message":"hello World"}

def json_to_numpy(jsondata:str, field:str):
    """json to numpy
    Args:
        jsondata (str): _description_

    Returns:
        _type_: _description_
    """
    as_dict = json.loads(jsondata)
    return np.array(as_dict[field])

def numpy_to_json(array):
    """numpy to json

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    return json.dumps({array.tolist()})

@app.post("/predict")
def infer(feature_vector:list)->list:
    """inference

    Args:
        input (str): _description_

    Returns:
        list: _description_
    """
    feature_vector = json.loads(feature_vector)
    feature_vector = json_to_numpy(feature_vector, field="feature_vector")
    predictions = model.predict(feature_vector)
    return json.dumps({"predictions":predictions.tolist()})

def load_model_and_encoder(path_to_model:str):
    """Load sklearn classifier(model) and encoder

    Args:
        path (_type_): path to object

    Returns:
        _type_: sklearn classifier, sklearn encoder
    """
    classifier, encoder = joblib.load(path_to_model) #load the model and encoder
    return classifier, encoder

def load_dataframe(path:str)->pd.DataFrame:
    """Load pandas dataframe

    Args:
        path (_type_): path to object

    Returns:
        _type_: dataframe
    """
    dataframe = pd.read_pickle(path)
    return dataframe

@app.post("/get_scenario")
def randomly_generate_scenario(n_examples:int=1)->tuple:
    """randomly generate a scenario

    Args:
        encoder (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        n (int, optional): _description_. Defaults to 1.

    Returns:
        tuple: _description_
    """
    n_features = encoder.n_features_in_

    feature_vector = np.random.randint(low=np.zeros(shape=(n_examples,n_features)),
                high=[len(encoder.categories_[i]) for i in range(len(encoder.categories_))],
                size=(n_examples,n_features),
                dtype=np.int32)
                
    scenario = encoder.inverse_transform(feature_vector)

    return  json.dumps({"feature_vector":feature_vector.tolist(),
        "scenario":scenario.tolist()})

@app.post("/get_historical_results")
def reverse_dfquery(example:str)->str:
    """_summary_

    Args:
        example (np.ndarray): _description_
        encoder (OrdinalEncoder): _description_
        dataframe (pd.DataFrame): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    example = json.loads(example)
    example = json_to_numpy(example, field="feature_vector")

    result=pd.DataFrame(columns=dataframe.columns)
    features = encoder.feature_names    
    for n in range(example.shape[0]):
        stacked = ""
        s =""
        scenario = encoder.inverse_transform(example[n,:].reshape(1,-1))[0]
        for i, condition in enumerate(scenario):
            if ' ' in features[i]:
                col = f'(`{features[i]}`=='
            else:
                col = f'({features[i]}=='

            if isinstance(condition, str):
                cond = f'"{condition}")'
            else:
                cond = f'{condition})'

            if i != encoder.n_features_in_-1:
                s += col + cond +' and '
            else:
                s += col + cond

        stacked += s        
        temp = dataframe.query(expr=stacked, inplace=False)
        result = pd.concat([result, temp])
    return result.to_json(orient="split")

@app.post("/get_historical_results_summary")
def check_results(predictions:list)->tuple:
    """Check Results against historical database

    Args:
        predictions (np.ndarray): predicted outcome
        dataframe (pd.DataFrame): historical data

    Returns:
        _type_: tuple(prediction_true_proportion, historical_true_proportion)
    """
    predictions = np.ndarray(predictions)
    historical_true_proportion = dataframe["Arrest Flag"].sum()/dataframe.size
    prediction_true_proportion = predictions.sum()/len(predictions)
    return prediction_true_proportion, historical_true_proportion

## start-up methods
model, encoder = load_model_and_encoder(PATH_TO_MODEL)
dataframe = load_dataframe(PATH_TO_DATA)
uvicorn.run(app)
