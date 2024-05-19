This is a model that predicts the winner of a ufc fight. It consist of an ensemble of four models (Neural network, XGBClassifier, RandomForestClassifier, GaussianNB) whose prediction's are used on a LogisticRegression meta learner for the final result. The model achieved 0.72 precision for 0.13 recall on the test set.

The dataset is from https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset

The ufc_winner_predictor.py file contains all the preprocessing steps i took on the original data, and the preprocessing function i used in the deployed model

The tryouts.py file has all the model building, hyperparameter tuning and evaluations i did 

The results.py contains all the necessary components (models, datasets, pipelines etc) for the model to work and the prediction function. It is the file used in deployment

The fighters_per_weightclass_list is a list of all the available fighters to use in the deployed model
The model uses a dataset (made in ufc ufc_winner_predictor.py file) containing these names and their stats. This dataset can be updated for future usage. As per now the stats are collected in 2021
