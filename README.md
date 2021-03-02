# disaster-response-pipeline
ETL and machine learning pipeline to analyse and categorise real messages sent during disaster events to help generate information for appropriate disaster relief agencies. This project also includes a web app which displays summary visualisations of the message data and contains a search bar where an emergency worker can input a new message and get classification results in several categories.

### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)
3. [Running Pipeline](#running)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

In order to run the ETL and ML pipeline code you will need the standard libraries included within the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## File Descriptions <a name="files"></a>

The repository is divided into three sections based on the functionality of the files - data, models, and app. The files listed below are available within each respective section of this repository.

Data:

- process_data.py - ETL script to read the disaster datasets, clean the data and generate and store it in an SQLite database. This script will output a SQLite database file.

- disaster_messages.csv - csv file containing text and medium information for messages sent during disaster events.

- disaster_categories.csv - csv file containing categorisation data for messages sent during disaster events.

Models:

- train_classifier.py - machine learning pipeline script that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the cleaned message data from process_data.py to predict classifications for 36 different categories (multi-output classification). This file also exports the trained model as a pickle file. 

App:

- run.py - script to build the back-end of the web app using the Flask framework.

- master.html - html file containing the front-end layout for the web app.

- go.html - html file containing the front-end layout for the search bar within the web app.

## Running Pipeline<a name="running"></a>

To run the pipeline, set up your database and model, and deploy the web app from within the project's root directory using the command line:

1. Run the process_data.py script specifying filepaths for loading the disaster message and category datasets and saving the SQLite database:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Run the train_classifier.py script specifying filepaths for loading the SQLite database and saving the trained model:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Deploy the web app:

cd app
python run.py

4. Go to http://0.0.0.0:3001/ to view web app

## Results<a name="results"></a>

Below are a few screenshots from the web app.

The first image shows the search bar where an emergency worker can input a new message and get classification results in several categories - the results from an example message are shown in the second image.

![search bar](https://github.com/greg-jones-code/disaster-response-pipeline/blob/main/images/search-bar.png)

![search bar results](https://github.com/greg-jones-code/disaster-response-pipeline/blob/main/images/search-bar-results.png)

The web app also contains three visualisations to display the distribution of message genres and the most and least frequent categories predicted from the messages within the dataset. Examples of these plots are shown below.

![distribution of message genres](https://github.com/greg-jones-code/disaster-response-pipeline/blob/main/images/distribution-of-message-genres-plot.png)

![most frequent categories](https://github.com/greg-jones-code/disaster-response-pipeline/blob/main/images/most-frequent-categories-plot.png)

![least frequent categories](https://github.com/greg-jones-code/disaster-response-pipeline/blob/main/images/least-frequent-categories-plot.png)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Figure Eight](https://www.figure-eight.com/) (now [Appen](https://appen.com/?ref=Welcome.AI)) for the data.