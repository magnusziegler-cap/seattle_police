from random import Random
from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
import json

import pandas as pd
import numpy as np

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

from pydantic import Field, BaseModel
from typing import Union


## Statics
PATH_TO_MODEL = 'models\\model_and_encoder.joblib'
PATH_TO_DATA = 'data\\police_stops.pickle'

app = FastAPI(title="Seattle Police What If Scenario Generator/Interpreter")

# Functions, Non-Endpoints
def load_model_and_encoder(path_to_model:str)-> tuple:
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

def infer(X:np.ndarray, model:Union[CategoricalNB, RandomForestClassifier]):
    predictions = model.predict(X)
    return predictions

def reverse_dfquery(example:np.ndarray, encoder:OrdinalEncoder, dataframe:pd.DataFrame, verbose=False)->pd.DataFrame:

    result=pd.DataFrame(columns=dataframe.columns)
    features = encoder.feature_names
    
    for n in range(example.shape[0]):
        
        stacked = ""
        s =""
        scenario = encoder.inverse_transform(example[n,:].reshape(1,-1))[0]

        for i, condition in enumerate(scenario):

            if (' ' in features[i]):
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
        
        if verbose:
            print("Query: ", stacked)
            
        temp = dataframe.query(expr=stacked, inplace=False)
        result = pd.concat([result, temp])
    
    return result

def check_results(predictions:np.ndarray, dataframe:pd.DataFrame)-> dict:
    if dataframe.size != 0:
        historical_true_proportion = dataframe["Arrest Flag"].sum()/dataframe.size
    else:
        historical_true_proportion = 0
    prediction_true_proportion = predictions.sum()/len(predictions)
    return {"Predicted True Proportion":prediction_true_proportion,
        "Historical True Proportion":historical_true_proportion}

def randomly_generate_scenario(n_examples:int, encoder:OrdinalEncoder)->tuple:
    """randomly generate a scenario

    Args:
        encoder (_type_): _description_
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

    return  feature_vector, scenario

def generate_scenario(feature_vector:np.ndarray, encoder:OrdinalEncoder)-> list:
    scenario = encoder.inverse_transform(feature_vector)
    # features = encoder.categories_
    # scenario=[]
    # for i in range(len(encoder.feature_names)):
    #     s=[]
    #     for n in range(feature_vector.shape[0]):
    #         condition = int(feature_vector[n,i])
    #         s.append(features[i][condition])
    #     scenario.append(s)
    return scenario

def _make_next_id()->int:
    """generate next id in sequence based of length of list

    Returns:
        int: id number
    """
    id =  max(Query.id for Query in Queries) + 1
    return id

## data model
class Query(BaseModel):
    """Data model for server

    Args:
        BaseModel (_type_): _description_
    """
    id:int = Field(default_factory=_make_next_id, alias="id")
    predictions: Union[list[bool], None] = None
    feature_vector: Union[list, None] = None
    features: Union[list, None] = None
    scenario: Union[list, None] = None
    historical_data: Union[dict, None] = None
    historical_results: Union[dict, None] = None

## Warm-start the query database with blank-ish query
Queries = [Query(
            id=0,
            predictions=None,
            feature_vector=None,
            features=None,
            scenario=None,
            historical_data=None,
            historical_results=None)
        ]

## Endpoints
@app.get("/queries")
async def queries():
    return Queries

@app.get("/queries/{id}")
async def query(id:int):
    if id > len(Queries):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        return Queries[id]

@app.post("/random_query", status_code=201)
async def add_query(query: Query, n_examples:int=1):
    feature_vector, scenario = randomly_generate_scenario(n_examples, encoder)
    features = encoder.feature_names
    predictions = infer(feature_vector, model)
    historical_data = reverse_dfquery(feature_vector, encoder, dataframe)
    historical_results = check_results(predictions, historical_data)

    query_rng = Query(
        predictions=predictions.tolist(),
        feature_vector=feature_vector.tolist(),
        features=features,
        scenario=scenario.tolist(),
        historical_data=historical_data.to_dict(),
        historical_results=historical_results
        )

    Queries.append(query_rng)
    return query_rng

#2,6,1,4,8,2,0,1,3
@app.post("/manual_query", status_code=201)
async def add_query_from_feature_vector(feature_vector:str):

    feature_vector = feature_vector.split(',')
    feature_vector = np.array([[int(f) for f in feature_vector]])

    predictions = infer(feature_vector, model)
    scenario = generate_scenario(feature_vector, encoder)
    features = encoder.feature_names
    historical_data = reverse_dfquery(feature_vector, encoder, dataframe)
    historical_results = check_results(predictions, historical_data)
    query = Query(
        predictions=predictions.tolist(),
        feature_vector=feature_vector.tolist(),
        features = features,
        scenario=scenario.tolist(),
        historical_data=historical_data.to_dict(),
        historical_results=historical_results
        )
    Queries.append(query)
    return query

@app.get("/encoding")
async def encoder_info():
    features = encoder.feature_names
    categories = encoder.categories_

    mapping = dict()
    for i, feature in enumerate(features):
        mapping[feature] = list(categories[i])

    return json.dumps(mapping)

## start-up methods
model, encoder = load_model_and_encoder(PATH_TO_MODEL)
dataframe = load_dataframe(PATH_TO_DATA)
uvicorn.run(app)
