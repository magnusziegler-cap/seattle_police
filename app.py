from subprocess import call
from dash import Dash, html, dcc, Input, Output, dash_table, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import joblib
import pickle
import requests

# Statics
SERVER_BASE = "127.0.0.1"
PORT = "8000"

app = Dash(__name__,
        title="What-if in Seattle",
        update_title="updating...")

scenario_creation_group = []


app.layout = html.Div(
    children = [
        html.Div(
            children=[
                html.H1(children="Seattle Police: 'Terry Stops' - What-if?"),
                html.P(children=[
                """This page can be used to generate a random scenario which the classifier
                can evaluate and determine whether or not situation is likely to end in an arrest.
                Following this, the database of previous events can be searched to determine whether
                or not that situation has happened in the past.
                Finally, users can select a scenario they are interested in and do the same what-if analysis."""]),
            ]
        ),
        html.Button(id="generate_scenario_randomly",
            children="Randomly Generate Scenario",
            n_clicks=0),
        html.Div(
            children=[
                dbc.Button(
                    "Open Scenario Details",
                    id="button_collapse_scenario",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    children=[
                        dbc.Card(
                            children=
                                [
                                    dbc.CardBody("Scenario:"),
                                    html.P(children=[],
                                        id="collapse_scenario_text"),
                                    html.P(children=[],
                                        id="collapse_prediction_text"),
                                ]
                        ),
                        scenario_creation_group,
                    ],
                    id ="collapse_scenario",
                    is_open=False,
                    )
            ]
        ),
        html.Div(
            children=[
                html.Table(id="table-historical-data")
            ]
        ),
        dcc.Store(id="dataframe", storage_type="session"),
        dcc.Store(id="scenario_feature_vector", storage_type="memory"),
        dcc.Store(id="scenario_feature_values", storage_type="memory"),
        dcc.Store(id="scenario_prediction", storage_type="memory"),
    ]
)

@app.callback(
    Output("collapse_scenario", "is_open"),
    [Input("button_collapse_scenario", "n_clicks")],
    [State("collapse_scenario", "is_open")]
)
def collapse(n_clicks, is_open)-> bool:
    """Collapses the scenario card

    Args:
        n (_type_): number of clicks. must be >1 for soemthing to happen, i.e. starts closed
        is_open (bool): is the card open

    Returns:
        _type_: boolean is_open
    """
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("policeStops-df", "data"),
    Input("scenario_feature_vector", "data")
)
def get_historical_data(scenario_feature_vector):
    endpoint="/get_historical_results"
    url="http://"+SERVER_BASE+":"+PORT+endpoint
    response = requests.post(url=url, timeout=5, data={"feature_vector":scenario_feature_vector})
    historical_df = pd.read_json(response.json())
    return historical_df


@app.callback(
    Output("table-historical-data", "children"),
    Input("policeStops-df", "data"),
    )
def update_datatable(df:pd.DataFrame):
    """Updates datatable html component

    Args:
        df (_type_): dataframe

    Returns:
        _type_: figure
    """
    df = pd.read_json(df, orient='split')

    fig = dash_table.DataTable(data=df.to_dict('records'),
        columns=[{"name": i, "id": i,'hideable':True} for i in df.columns],
        style_cell=dict(textAlign='left'),
        cell_selectable=True,
        row_selectable='multi',
        sort_action='native',
        hidden_columns=[],
        filter_action='native',)

    return fig

@app.callback(
    Output("scenario_feature_values", "data"),
    Output("scenario_feature_vector", 'data'),
    Input("generate_scenario_randomly", "n_clicks")
)
def get_random_scenario(n_clicks):
    if n_clicks:
        endpoint="/get_scenario"
        url="http://"+SERVER_BASE+":"+PORT+endpoint
        response = requests.post(url=url, timeout=5, data={"n_examples":1})
        status_code = response.status_code
        content = json.loads(response.json())
        return content["scenario"], content["feature_vector"]
    else:
        return None, None

@app.callback(
    Output("collapse_scenario_text", "children"),
    Input("scenario_feature_values", "data")
)
def print_scenario(scenario):
    if scenario is not None:
        output = str(scenario)
        return output

@app.callback(

    Output("scenario_prediction", 'data'),
    Input("scenario_feature_vector", "data")
)
def get_inference_result(scenario_feature_vector):
    endpoint="/predict"
    url="http://"+SERVER_BASE+":"+PORT+endpoint
    response = requests.post(url=url, timeout=5, data={"feature_vector":json.dumps([scenario_feature_vector])})
    status_code = response.status_code
    content = json.loads(response.json())
    predictions = True
    return str(status_code), predictions

@app.callback(
    Output("collapse_prediction_text", "children"),
    Input("scenario_prediction", 'data')
)
def print_scenario_prediction(prediction):
    if prediction is not None:
        output = str(*[prediction])
        return output

def load_dataframe(path_to_data:str='data\\police_stops.pickle'):
    """Load Dataframe from given path

    Args:
        path_to_data (str): path to dataframe

    Returns:
        _type_: dataframe
    """
    dataframe = pd.read_pickle(path_to_data)

    return dataframe.to_json(orient="split")

if __name__ == '__main__':
    app.run_server(debug=True)

