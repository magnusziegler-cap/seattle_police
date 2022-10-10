# Seattle Police Terry Stops
Magnus Ziegler
magnus.ziegler@capgemini.com
## Overview
This repository contains a training project used to explore the use of the FastAPI package for basic client-server relationships in combination with Dash for a basic dashboard/UI.

The [Seattle Police dataset](https://www.kaggle.com/datasets/city-of-seattle/seattle-terry-stops) represents records of police reported stops under Terry v. Ohio, 392 U.S. 1 (1968). Each row represents a unique stop.

Each record contains perceived demographics of the subject, as reported by the officer making the stop and officer demographics as reported to the Seattle Police Department, for employment purposes.

Where available, data elements from the associated Computer Aided Dispatch (CAD) event (e.g. Call Type, Initial Call Type, Final Call Type) are included.

## Specifics

+ /notebooks: project notebook for exploration and basic analysis
+ /client: dash-app that acts as the client
+ /server: fastAPI server that performs, stores, and reports the results of queries on the dataset
+ /data: original archive zip file, pickled pandas dataframe, schema, and csv
+ /models: saved scikitlearn-derived model and encoder in joblib format

To start the project, run server/main.py to start the server, and then client/app.py to start the UI client.

## Improvements

+ Improved UI/UX in the dash app, maybe with visuals to make the scenario generation more interesting
+ report the prediction probability from the model
+ report query stats
+ re-organize server code to keep only the endpoints in main.py, and shift support functions to a secondary module

