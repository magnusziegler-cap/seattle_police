from dash import Dash, html, dcc, Input, Output, dash_table, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
import requests
import pprint as pp
import joblib

# Statics
SERVER_BASE = "127.0.0.1"
PORT = "8000"
PATH_TO_MODEL = '.\\models\\model_and_encoder.joblib'

app = Dash(__name__,
        title="Seattle Police What If Scenario Viewer",
        update_title="updating...",
        external_stylesheets=[dbc.themes.BOOTSTRAP])

collapse_scenario = html.Div(
    children=[
        dbc.Button(
            "Open Scenario Details",
            id="button_collapse_scenario",
            n_clicks=0,
        ),
        dbc.Collapse(
            children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H3("Scenario"),
                        html.Div(children=[
                            dbc.Switch(
                                label="Display Manually Created or Automatic Scenario",
                                id="scenario_switch",
                                value=False),
                        ]),
                        dbc.Table(id="collapse_scenario_table"),
                        html.H3("Arrest Prediction"),
                        html.P(id="collapse_prediction_text"),
                        html.H3("Historical Data"),
                        html.P("If this scenario occured in the database, the previous occurences will be shown below."),
                        dbc.Table(id="table-historical-data")
                        ]),
                    ]
                ),
            ],
            id ="collapse_scenario",
            is_open=False,
            )
    ],
    style={"display": "flex", "flexWrap": "wrap", "width":"80%"})

collapse_encoding = html.Div(
    children=[
        dbc.Button(id="get_encoding_button",
            children="Open Encoding Details",
            n_clicks=0),
        dbc.Collapse(
            children=[
                dbc.Card(children=[
                    dbc.CardBody(
                        children=[
                            html.P(children=[],
                                id="collapse_encoding_text"),
                        ]),
                ]),
                ],
            id = 'collapse_encoding',
            is_open=False,
        )
    ],
    style={"display": "flex", "flexWrap": "wrap", "width":"80%"}
)

def create_creation_dropdowns():
    _,  encoder = joblib.load(PATH_TO_MODEL)

    features = encoder.feature_names
    components = []
    for i,feature in enumerate(features):
        sanitized_feature_name = feature.replace(' ','_')
        c = dbc.Select(
            placeholder=feature,
            options=[{"label":str(category),"value":enc} for enc,category in enumerate(encoder.categories_[i])],
            id={"id":f'create_select_{sanitized_feature_name}',
                "type":'create-scenario-dropdown'}
        )
        components.append(c)
    
    group = html.Div(children=components, style={"display": "flex", "flexWrap": "wrap", "width":"80%"},)
    return group

creation_controls=create_creation_dropdowns()

collapse_create = html.Div(
    children=[
        dbc.Button(id="button_create_collapse",
            children="Open Manual Scenario Generation Tool",
            n_clicks=0),
        dbc.Collapse(
            children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H3("Create your own scenario to evaluate:"),
                        html.P(children=[],
                            id="collapse_create_text"),
                        creation_controls,
                        dbc.Button(id="button_create_evaluate",
                            children="Evaluate Created Scenario",
                            n_clicks=0),
                        html.P(children=[],
                            id="collapse_create_results_text"),
                    ]),
                ]),
            ],
            id = 'collapse_create',
            is_open=False,
        ),
    ],
    style={"display": "flex", "flexWrap": "wrap", "width":"80%"}
)

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
            ],
            style={"display": "flex", "flexWrap": "wrap", "width":"80%"}
        ),
        html.Div(
            children=[
                dbc.Button(id="generate_scenario_randomly",
                    children="Randomly Generate Scenario",
                    n_clicks=0,
                    style={"width":"40%", "display":'inline-block'},
                    ),
                dbc.Input(
                    placeholder="Number of Scenarios to generate",
                    type="number",
                    min=1,
                    max=1000,
                    id="num_examples_input",
                    list="1,5,10,20",
                    style={"width":"25%", "display":'inline-block'}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "width":"80%"}
        ),
        collapse_encoding,
        collapse_create,
        collapse_scenario,
        dcc.Store(id="scenario-auto", storage_type="memory"),
        dcc.Store(id="scenario-manual", storage_type="memory"),
        dcc.Store(id="encoding", storage_type="memory"),
    ]
)

@app.callback(
    Output("collapse_create", "is_open"),
    [Input("button_create_collapse", "n_clicks")],
    [State("collapse_create", "is_open")]
)
def toggle_collapse_create(n_clicks, is_open):
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
    Output("collapse_encoding", "is_open"),
    [Input("get_encoding_button", "n_clicks")],
    [State("collapse_encoding", "is_open")]
)
def toggle_collapse_encoding(n_clicks, is_open):
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
    Output("collapse_scenario", "is_open"),
    [Input("button_collapse_scenario", "n_clicks")],
    [State("collapse_scenario", "is_open")],
)
def toggle_collapse_scenario(n_clicks, is_open):
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
    Output("encoding", "data"),
    Input("get_encoding_button", "n_clicks"),
    prevent_initial_callback=True,
)
def get_encoder_details(n_clicks):
    if n_clicks:
        endpoint="/encoding"
        url="http://"+SERVER_BASE+":"+PORT+endpoint
        response = requests.get(url=url, timeout=5)
        mapping = json.loads(response.json())

    return mapping

@app.callback(
    Output("collapse_encoding_text", "children"),
    Input("encoding", "data"),
)
def print_encoding_details(mapping)->str:

    return f'Encoding Features and categories: \n {pp.pformat(mapping)}'

@app.callback(
    Output("scenario-auto", "data"),
    [Input("generate_scenario_randomly", "n_clicks"), Input("num_examples_input", "value")],
    prevent_initial_callback=True,
)
def query_random_scenario(n_clicks, n_examples):
    if n_clicks:
        endpoint="/random_query"
        url="http://"+SERVER_BASE+":"+PORT+endpoint
        payload = {"n_examples":n_examples}
        response = requests.post(url=url, timeout=30, params=payload, json=payload)
        scenario = response.json()

        return scenario

@app.callback(
    Output("table-historical-data", "children"),
    [Input("scenario-manual", "data"), Input("scenario-auto", "data")],
    State("scenario_switch","value"),
    prevent_initial_callback=True,
    )
def update_datatable(scenario_manual, scenario_auto, switch_value):
    """Updates datatable html component

    Args:
        df (_type_): dataframe

    Returns:
        _type_: figure
    """
    if switch_value:
        scenario = scenario_manual
    else:
        scenario = scenario_auto

    df = pd.DataFrame(scenario["historical_data"])

    fig = dash_table.DataTable(data=df.to_dict('records'),
        columns=[{"name": i, "id": i,'hideable':True} for i in df.columns],
        style_cell=dict(textAlign='left'),
        cell_selectable=True,
        row_selectable='multi',
        sort_action='native',
        hidden_columns=[],
        filter_action='native',
        style_table={'overflowX': 'auto', 'maxWidth':"1080px",
            "overflowY":'auto', "maxHeight":"720px"},)

    return fig

@app.callback(
    Output("collapse_scenario_table", "children"),
    [Input("scenario-manual", "data"), Input("scenario-auto", "data")],
    State("scenario_switch","value"),
    prevent_initial_callback=True,
)
def print_scenario(scenario_manual, scenario_auto, switch_value):
    if switch_value:
        scenario = scenario_manual
    else:
        scenario = scenario_auto

    df = pd.DataFrame.from_dict(scenario["scenario"])
    df.columns = (scenario["features"])

    fig = dash_table.DataTable(data=df.to_dict('records'),
        columns=[{"name": i, "id": i,'hideable':True} for i in df.columns],
        style_cell=dict(textAlign='left'),
        cell_selectable=True,
        row_selectable='multi',
        sort_action='native',
        hidden_columns=[],
        filter_action='native',
        style_table={'overflowX': 'auto', 'maxWidth':"1080px",
            "overflowY":'auto', "maxHeight":"720px"})

    return fig

@app.callback(
    Output("collapse_prediction_text", "children"),
    [Input("scenario-manual", "data"), Input("scenario-auto", "data")],
    State("scenario_switch","value"),
    prevent_initial_callback=True,
)
def print_prediction(scenario_manual, scenario_auto, switch_value):
    """_summary_

    Args:
        scenario_manual (_type_): _description_
        scenario_auto (_type_): _description_
        switch_value (_type_): _description_

    Returns:
        _type_: _description_
    """
    if switch_value:
        scenario = scenario_manual
    else:
        scenario = scenario_auto

    return f'Predicted Arrest(s): {scenario["predictions"]}'

@app.callback(
    Output("scenario-manual", "data"),
    [Input("button_create_evaluate","n_clicks"),
    Input({"type":"create-scenario-dropdown", "id":ALL}, 'value')],
    prevent_initial_call=True
)
def query_user_scenario(n_clicks, values):
    """Query server with user generated scenario

    Args:
        n_clicks (_type_): evaluate button click counter
        values (_type_): values from creation menus

    Returns:
        _type_: scenario
    """
    if n_clicks:
        endpoint="/manual_query"
        url="http://"+SERVER_BASE+":"+PORT+endpoint
        fv = str(values).replace("[",'').replace("]",'').replace(" ","").replace("'","")
        payload = {"feature_vector":fv}
        response = requests.post(url=url, timeout=10, params=payload, json=payload)
        scenario = response.json()

        st = f" Prediction: {scenario['predictions']}"

        return scenario

if __name__ == '__main__':
    app.run_server(debug=False)
