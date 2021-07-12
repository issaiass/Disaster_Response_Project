# Disaster Response Pipeline Project

## Results Screenshot

<p align="center">
<b>Figure 1 - Disaster Response Predictions</b>
<img src = "imgs/disaster_classifier_app.PNG?raw=true" width="75%"/>
</p>

<p align="center">
<b>Figure 2 - Disaster Respose Dashboard</b>
<img src = "imgs/disaster_dashboard.PNG?raw=true" width="75%"/>
</p>

## Installation

Please see "part I" of "Launch Instructions" below to install the dependencies associated to Python, basically they are:
- pandas
- json
- plotly
- nltk
- flask
- sqlalchemy
- numpy
- re
- pickle
- sklearn

## Description or Motivation of the Project

This Udacity project provides tweets and text messages from real disasters.  The first task is to use jupyter notebook to prepare a ETL pipeline and then, again, using jupyter notebooks prepare a Machine Learning pipeline to build a supervised learing model using scikit-learn.

The main idea is that when a disaster occurs we receive millions of messages by social media, but the organizations has less time to process those messages because we do not know how many of them are more critical than others.

Then, because each organization get a part of the problem is really important to understand the context of the message and translate the "help message" to the correct organization.

## File Organization and Content
I. ETL Pipeline Preparation.ipynb: Contains the preprocessing of the data and conversion of the messages to a database file.

II. ML Pipeline Preparation.ipynb:  Contains the build of a multiclass output classifier and evaluation metrics

III. APP Folder

`app/templates`:  Contains the webpages neccesary to render.

`app/run.py`:  Run the server, render the page and waits for the model input parameters

IV.  Data Folder

`data/disaster_messages.csv`

`data/disaster_categories.csv`

The files above contains tweet of disaster information that will be used not only to preprocess but train a multiclass output classifier.

`data/process_data.py`

As it states, can do a ETL pipeline for clean/drop, extract and store in a database.

V.  Models Folder

`train_classifier.py`:  The machine learning pipeline that load the database, trains the model, evaluate and saves the best model dumping into serialized file.

## Launch Instructions:
I. Install dependencies if you need them using:

    pip install -r requirements.txt

II. Run the app

If you want to run in the local server:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

If you want to run in the udacity server:
1. Follow steps 1 to 3.

2. On a terminal get the environment variables using the command:
    
    `env | grep WORK`

3. Get the workspace doman and workspace id values, in my case was:

    `WORKSPACEDOMAIN=udacity-student-workspaces.com`
    `WORKSPACEID=view6914b2f4`

4. Go to `https://WORKSPACEID-3001.WORKSPACEDOMAIN/` but replace the workspace id and workspace domain with your values.