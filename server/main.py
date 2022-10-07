from typing import Union

import joblib
import json
import pickle

import pandas as pd
import numpy as np

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import Field, BaseModel

import datetime

## Statics
PATH_TO_MODEL = '.\\models\\model_and_encoder.joblib'
PATH_TO_DATA = '.\\data\\police_stops.pickle'

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

def load_dataframe(path_to_data:str)->pd.DataFrame:
    """Load pandas dataframe from pickle

    Args:
        path (_type_): path to object

    Returns:
        _type_: dataframe
    """
    dataframe = pd.read_pickle(path_to_data)
    return dataframe

def infer(X:np.ndarray, model:Union[CategoricalNB, RandomForestClassifier])-> np.ndarray:
    """ Predict label

    Args:
        X (np.ndarray): Input feature vector, or batch of
        model (Union[CategoricalNB, RandomForestClassifier]): SciKitLearn Model

    Returns:
        np.ndarray: predictions
    """
    predictions = model.predict(X)
    return predictions

def reverse_dfquery(example:np.ndarray, encoder:OrdinalEncoder, dataframe:pd.DataFrame, verbose=False)->pd.DataFrame:
    """Reverse Query the Dataframe

    Args:
        example (np.ndarray): Query 
        encoder (OrdinalEncoder): Feature Encoder
        dataframe (pd.DataFrame): Relevant Dataframe
        verbose (bool, optional): Print query to console. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe representing the result of the query
    """
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
    """ Compares predicted results against historical data
    Calculates Predicted True Proportion and historical true proportion,
    and returns those as a dict.

    Args:
        predictions (np.ndarray): _description_
        dataframe (pd.DataFrame): _description_

    Returns:
        dict: {Predicted True Proportion, Historical True Proportion}
    """
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
    """Perform inverse transform with Ordinal Encoder to generate scenario 
    Args:
        feature_vector (np.ndarray): Feature vector
        encoder (OrdinalEncoder): Ordinal Encoder

    Returns:
        list: list of strings detailing the scenario described by the feature vector
    """
    scenario = encoder.inverse_transform(feature_vector)
    return scenario

def _make_next_id()->int:
    """generate next id in sequence based of length of list

    Returns:
        int: id number
    """
    index =  max(Query.id for Query in Queries) + 1
    return index

def save_database(queries) -> None:
    with open('seattle_what_ifs.pickle', 'wb') as file_handle:
        pickle.dump(queries, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

def update_database(query, filename) -> None:
    with open(filename, 'a') as file_handle:
        file_handle.write(str(query))

def create_database():
    timestamp = datetime.datetime.strftime(datetime.datetime.now(),format="%Y%m%d_%H%M")
    filename = f'seattle_police_whatif_queries_{timestamp}.txt)'
    with open(filename, 'x+') as file_handle:
        file_handle.write("")
    return filename

## data model
class Query(BaseModel):
    """Data model for server

    Args:
        id:int = identifier number
        predictions: list of predicted arrests
        feature_vector: feature vector for the query
        features: string list of the features involved
        scenario: string list of the feature specific labels
        historical_data: dict of previous data, formatted from pandas.to_dict()
        historical_results: dict of summary data from historical_data dataframe
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
async def queries()->list[Query]:
    """Returns the list of Queries

    Returns:
        list[Query]: Query Database
    """
    return Queries

@app.get("/queries/{id}")
async def query(id:int)->Query:
    """ Retrieves a specific Query

    Args:
        id (int): requested query ID

    Raises:
        HTTPException: If requested ID does not exist in database.

    Returns:
        Query: Query results
    """
    if id > len(Queries):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        return Queries[id]

@app.post("/random_query", status_code=201)
async def add_query(query:Query, n_examples:int=1)->Query:
    """Query from randomly generated feture vector

    Args:
        n_examples (int): number of examples to generate

    Returns:
        Query: Query results
    """
    feature_vector, scenario = randomly_generate_scenario(n_examples, encoder)
    features = encoder.feature_names
    predictions = infer(feature_vector, model)
    historical_data = reverse_dfquery(feature_vector, encoder, dataframe)
    historical_results = check_results(predictions, historical_data)

    query = Query(
        predictions=predictions.tolist(),
        feature_vector=feature_vector.tolist(),
        features=features,
        scenario=scenario.tolist(),
        historical_data=historical_data.to_dict(),
        historical_results=historical_results
        )

    Queries.append(query)

    update_database(query, FILENAME)

    return query

@app.post("/manual_query", status_code=201)
async def add_query_from_feature_vector(feature_vector:str)->Query:
    """Query from manually generated feture vector
    testable query:2,6,1,4,8,2,0,1,3
    Args:
        feature_vector (str): feature vector of correct length, comma separated

    Returns:
        Query: Query results
    """

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

    update_database(query, FILENAME)

    return query

@app.get("/encoding")
async def encoder_info()->str:
    """ Returns encoder information, detailing the mapping.
    Features->categories, and the ordinal encoding

    Returns:
        str: _description_
    """
    features = encoder.feature_names
    categories = encoder.categories_

    mapping = {}
    for i, feature in enumerate(features):
        mapping[feature] = list(categories[i])

    return json.dumps(mapping)

## start-up server methods
model, encoder = load_model_and_encoder(PATH_TO_MODEL)
dataframe = load_dataframe(PATH_TO_DATA)
FILENAME = create_database()
uvicorn.run(app)
