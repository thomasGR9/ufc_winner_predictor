# UFC Fight Outcome Predictor

This project is a machine learning model that predicts the winner of a UFC fight. It consists of an ensemble of four models (Neural Network, XGBClassifier, RandomForestClassifier, GaussianNB) whose predictions are used by a LogisticRegression meta-learner for the final result. The model achieved a precision of 0.72 and a recall of 0.13 on the test set.

## Dataset

The dataset used is from [Kaggle's Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset).

## Project Structure

- **ufc_winner_predictor.py**: Contains all the preprocessing steps applied to the original data, the preprocessing function used in the deployed model and the fighters_per_weightclass_list. The preprocessing includes: Feature engineering, imputations, making stratified train and test sets, outlier detection, scaling, PCA.
- **tryouts.py**: Includes all model building, hyperparameter tuning, and evaluations. These invovle: The keras tuner library for NN hyperparameter tuning, tranfer learning techniques to use the trained model on other features, bayesian hyperparameter search for RFC, XGB and meta learner and some costum functions to evaluate precision and recall of the models for both classes.
- **results.py**: Contains all necessary components (models, datasets, pipelines, etc.) for the model to work and the prediction function. This file is used in deployment.
- **fighters_per_weightclass_list**: A list of all available fighters to use in the deployed model. The model uses a dataset (created in ufc_winner_predictor.py) containing these names and their stats. This dataset can be updated for future usage; currently, the stats are collected up to 2021.

## Deployment

I created a website where you can test the model: [UFC Winner Predictor](https://ufcwinnerpredictor.online).

The following files make up the site:
- **index.html**: HTML file for the site structure.
- **styles.css**: CSS file for styling the site.
- **script.js**: JavaScript file for site functionality.

I also made an API using FastAPI that utilizes the final function in results.py to return the fight prediction to the site. The API files are in a Docker container and uploaded to a cloud host.

### Disclaimer

Please note that the site might not work if the API is shut down to save hosting costs. If you are interested in seeing the deployment in action, do not hesitate to contact me. My contact information is available on my GitHub account.
