import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from skopt import space
from functools import partial
from skopt import gp_minimize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import random
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import keras_tuner as kt
from sklearn.calibration import CalibratedClassifierCV
dataset_choice = "same_weight_class_ratios" #Choose between "same_weight_class_ratios" or "same_y_ratios"

def training_test_sets(dataset_ch):
    if dataset_ch == "same_weight_class_ratios":
        X_train_full_same_weight_class_ratios= pd.read_csv('./X_train_concat.csv', dtype=np.float64)
        X_test_full_same_weight_class_ratios= pd.read_csv('./X_test_concat.csv', dtype=np.float64)
        y_train_same_weight_class_ratios= pd.read_csv('./y_train.csv', dtype=np.float64)
        y_test_same_weight_class_ratios= pd.read_csv('./y_test.csv', dtype=np.float64)
        
        X_train_without_pca_full_same_weight_class_ratios= pd.read_csv('./X_train_num_scaled.csv', dtype=np.float64)
        X_test_without_pca_full_same_weight_class_ratios= pd.read_csv('./X_test_num_scaled.csv', dtype=np.float64)
        
        y_test_fair_full = y_test_same_weight_class_ratios.copy()
        X_test_fair_full = X_test_full_same_weight_class_ratios.copy()
        X_test_fair_without_pca_full = X_test_without_pca_full_same_weight_class_ratios.copy()
        #we drop instances that the winner is 0 from the start to the finish until we achieve the same ratio
        for index in y_test_fair_full.index:
            ratio = ((y_test_fair_full == 1).sum() / (y_test_fair_full == 0).sum())[0]
            if (ratio < 1):
                if (y_test_fair_full.loc[index].values[0] == 0):
                    y_test_fair_full.drop(index, inplace=True)
                    X_test_fair_full.drop(index, inplace=True)
                    X_test_fair_without_pca_full.drop(index, inplace=True)
                    
        y_test_fair_full.reset_index(drop=True, inplace=True)
        X_test_fair_full.reset_index(drop=True, inplace=True)
        X_test_fair_without_pca_full.reset_index(drop=True, inplace=True)

        
        ind_list=[i for i in range(len(y_test_fair_full))]
        ind_list_rd = random.shuffle(ind_list)
        #shuffling the indexes randomly
        y_test_fair_full = y_test_fair_full.iloc[ind_list].reset_index(drop=True)        
        X_test_fair_full = X_test_fair_full.iloc[ind_list].reset_index(drop=True)
        X_test_fair_without_pca_full = X_test_fair_without_pca_full.iloc[ind_list].reset_index(drop=True)
                            
        y_test_full = y_test_same_weight_class_ratios.copy()
        X_test_full = X_test_full_same_weight_class_ratios.copy()
        y_train_full = y_train_same_weight_class_ratios.copy()
        X_train_full = X_train_full_same_weight_class_ratios.copy()

        X_train_without_pca_full = X_train_without_pca_full_same_weight_class_ratios.copy()  
        X_test_without_pca_full = X_test_without_pca_full_same_weight_class_ratios.copy()
    
    if dataset_ch == "same_y_ratios":
        #same as the other
        X_train_full_same_y_ratios = pd.read_csv('./X_train_concat_branch.csv', dtype=np.float64)
        X_test_full_same_y_ratios= pd.read_csv('./X_test_concat_branch.csv', dtype=np.float64)
        y_train_same_y_ratios= pd.read_csv('./y_train_branch.csv', dtype=np.float64)
        y_test_same_y_ratios= pd.read_csv('./y_test_branch.csv', dtype=np.float64)

        X_train_without_pca_full_same_y_ratios= pd.read_csv('./X_train_num_scaled_branch.csv', dtype=np.float64)
        X_test_without_pca_full_same_y_ratios= pd.read_csv('./X_test_num_scaled_branch.csv', dtype=np.float64)
        
        y_test_fair_full = y_test_same_y_ratios.copy()
        X_test_fair_full = X_test_full_same_y_ratios.copy()
        X_test_fair_without_pca_full = X_test_without_pca_full_same_y_ratios.copy()

        for index in y_test_fair_full.index:
            ratio = ((y_test_fair_full == 1).sum() / (y_test_fair_full == 0).sum())[0]
            if (ratio < 1):
                if (y_test_fair_full.loc[index].values[0] == 0):
                    y_test_fair_full.drop(index, inplace=True)
                    X_test_fair_full.drop(index, inplace=True)
                    X_test_fair_without_pca_full.drop(index, inplace=True)

        y_test_fair_full.reset_index(drop=True, inplace=True)
        X_test_fair_full.reset_index(drop=True, inplace=True)
        X_test_fair_without_pca_full.reset_index(drop=True, inplace=True)

        
        ind_list=[i for i in range(len(y_test_fair_full))]
        ind_list_rd = random.shuffle(ind_list)
        
        y_test_fair_full = y_test_fair_full.iloc[ind_list].reset_index(drop=True)        
        X_test_fair_full = X_test_fair_full.iloc[ind_list].reset_index(drop=True)
        X_test_fair_without_pca_full = X_test_fair_without_pca_full.iloc[ind_list].reset_index(drop=True)

               
        y_test_full = y_test_same_y_ratios.copy()
        X_test_full = X_test_full_same_y_ratios.copy()
        y_train_full = y_train_same_y_ratios.copy()
        X_train_full = X_train_full_same_y_ratios.copy()

        X_train_without_pca_full = X_train_without_pca_full_same_y_ratios.copy()
        X_test_without_pca_full = X_test_without_pca_full_same_y_ratios.copy()
    return y_test_full, X_test_full, y_train_full, X_train_full, X_train_without_pca_full, X_test_without_pca_full, X_test_fair_full, X_test_fair_without_pca_full, y_test_fair_full
#function to easily choose between the two different kind of datasets we have.

y_test_full, X_test_full, y_train_full, X_train_full, X_train_without_pca_full, X_test_without_pca_full, X_test_fair_full, X_test_fair_without_pca_full, y_test_fair_full = training_test_sets(dataset_choice)






#X with pca and cat
X_train_pca_cat = X_train_full[:3500].select_dtypes(np.float64)
X_valid_pca_cat = X_train_full[3500:].select_dtypes(np.float64)
X_test_pca_cat = X_test_full.copy()

#X with pca without cat
X_train_pca_full = X_train_full.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_train_pca = X_train_pca_cat[:3500].drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_valid_pca = X_valid_pca_cat.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_test_pca = X_test_pca_cat.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_test_fair_pca = X_test_fair_full.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)


#X without pca without cat
X_train_without_pca = X_train_without_pca_full[:3500].select_dtypes(np.float64)
X_valid_without_pca = X_train_without_pca_full[3500:].select_dtypes(np.float64)
X_test_without_pca = X_test_without_pca_full.copy().select_dtypes(np.float64)
#
y_train_final = y_train_full[:3500].select_dtypes(np.float64)['Winner']
y_valid_final = y_train_full[3500:].select_dtypes(np.float64)['Winner']
y_test_final = y_test_full.copy()['Winner']



(y_test_fair_full == 0).sum() / (y_test_fair_full == 1).sum()
(y_test_final == 0).sum() / (y_test_final == 1).sum()

#Here i tried to combat the imbalance of the winner == 0 instances
#The way is to use the pre-pca dataset and change the values of the features of some data such that B --> R , R --> B.The diff columns will be equal to (-1)*diff_column.Then for that data we will swamp the zero target to one until we got equal amounts of each in the training set.
#This can be done because it shouldn't matter what corner the winner happened to be placed.
'''
(y_train_full == 1).sum() / (y_train_full == 0).sum()
(y_test_full == 1).sum() / (y_test_full == 0).sum()
#so we must change N data point from 0 to 1 where N is
N = ((y_train_full == 0).sum() - (y_train_full == 1).sum()) / 2
X_train_pca_full_winner_zero = X_train_pca_full.copy()
indexes = []
for index in y_train_full.index:
    if y_train_full.iloc[index, 0] == 0:
        indexes.append(index)
X_train_pca_full_winner_zero = X_train_pca_full_winner_zero.iloc[indexes]
X_train_pca_full_winner_zero
#training set with only winners = 0




#i will use GaussianMixtures to cluster the zero winners training set.
bmg = BayesianGaussianMixture(n_components=20, n_init=10, random_state=42)
bmg.fit(X_train_pca_full)
bmg.weights_.round(2)


bic_score = []
aic_score = []

for i in range(1, 20, 1):
    gm = GaussianMixture(n_components=i, n_init=10)
    gm.fit(X_train_pca_full_winner_zero)
    bic_score.append(gm.bic(X_train_pca_full_winner_zero).round(2))
    aic_score.append(gm.aic(X_train_pca_full_winner_zero).round(2))
    
bic_score
aic_score
#it seems that the 10 clusters give the best result

gm = GaussianMixture(n_components=10, n_init=10)
gm.fit(X_train_pca_full_winner_zero)
weights = []
for i in range(len(gm.weights_)):
    weights.append(gm.weights_[i])
    
weights


#i will try to change some winners 0 --> 1 in a way that keeps the same ratios in the fitted clusters.
insances_per_cluster = []
for i in range(len(weights)):
    insances_per_cluster.append(int(N * weights[i]))

insances_per_cluster
#how many instances i should change from each cluster

gaus_preds = gm.predict(X_train_pca_full_winner_zero)


gaus_preds_list = gaus_preds.tolist()
[i for i, n in enumerate(gaus_preds_list) if n == 0]
#this returns a list of all the indexes of the instances that are in cluster 0

indexes_from_clusters = []

for cluster in range(10):
    indexes_from_clusters.append(random.choices([i for i, n in enumerate(gaus_preds_list) if n == cluster], k=insances_per_cluster[cluster]))

indexes_from_clusters[0]
#is a list of lists.List n has (instances per cluster[n]) number of instances (indexes) picked randomly from cluster n

original_intexes_from_clusters = []
for cluster in range(10):
    original_intexes_from_clusters.append(X_train_pca_full_winner_zero.reset_index().iloc[indexes_from_clusters[cluster]]['index'].tolist())

X_train_pca_full_winner_zero.reset_index().iloc[indexes_from_clusters[0]]['index'].tolist()

original_intexes_from_clusters  
#the same list of lists but with the original indexes of the dataset
#these are the indexes of the datapoints that we will change from 0 --> 1. That way the ratios of points in each cluster of the fitted gaussian mixtures will remain the same.
intexes_for_change = []
for i in range(10):
    intexes_for_change = intexes_for_change + original_intexes_from_clusters[i] 
#all lists in one
len(intexes_for_change)


intexes_for_change_no_dublicates = [] 
duplist = [] 
for i in intexes_for_change:
    if i not in intexes_for_change_no_dublicates:
        intexes_for_change_no_dublicates.append(i)
    else:
        duplist.append(i)

len(intexes_for_change_no_dublicates)
(len(duplist) / len(intexes_for_change)) * 100
#only 4.6% of the indexes are duplicates, that shouldn't be a problem
len(intexes_for_change_no_dublicates)
#this is the intexes_for_change list but without dublicates

intexes_for_change_no_dublicates.sort()
intexes_for_change_no_dublicates
#sorted indexes

X_train_without_pca_full.iloc[intexes_for_change_no_dublicates]
#these instances will be changed from 0 --> 1
for i in intexes_for_change_no_dublicates:
    if y_train_full.iloc[i, 0] == 1:
        print("problemo")


X_train_for_change = X_train_without_pca_full.copy()
X_train_for_change = X_train_for_change.iloc[intexes_for_change_no_dublicates]
X_train_for_change
cols_for_B = ['B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_win_by_Decision_Split', 'B_Height_cms', 'B_Reach_cms', 'B_age', 'sqrt_B_avg_SIG_STR_landed', 'sqrt_B_avg_TD_landed', 'sqrt_B_longest_win_streak', 'sqrt_B_total_rounds_fought', 'sqrt_B_wins', 'cbrt_B_current_win_streak', 'cbrt_B_avg_SUB_ATT', 'cbrt_B_losses']
cols_for_R = ['R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_win_by_Decision_Split', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'sqrt_R_avg_SIG_STR_landed', 'sqrt_R_avg_TD_landed', 'sqrt_R_longest_win_streak', 'sqrt_R_total_rounds_fought', 'sqrt_R_wins', 'cbrt_R_current_win_streak', 'cbrt_R_avg_SUB_ATT', 'cbrt_R_losses']
diff_cols = ['win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'Rank_dif']

dict_B_to_R = {}
for i in range(len(cols_for_B)):
    dict_B_to_R[cols_for_B[i]] = cols_for_R[i]
dict_B_to_R

dict_R_to_B = {}
for i in range(len(cols_for_B)):
    dict_R_to_B[cols_for_R[i]] = cols_for_B[i]
dict_R_to_B


X_train_for_change_B = X_train_for_change[cols_for_B]
X_train_for_change_B.rename(columns=dict_B_to_R, inplace=True)
X_train_for_change_B
#will rename the columns cols_for_B of the dataset to the corresponding cols_for_R columns

X_train_for_change_R = X_train_for_change[cols_for_R]
X_train_for_change_R.rename(columns=dict_R_to_B, inplace=True)
X_train_for_change_R
#the same but for R --> B

X_train_for_change_diff = X_train_for_change[diff_cols]
X_train_for_change_diff = -X_train_for_change_diff
X_train_for_change_diff
#Will multiply the diff_cols by (-1).This is because we made the diff cols as the blue corner - red corner values.Now that we want to reverse them we should multiply them by (-1)

y_train_for_change = y_train_full.copy()
y_train_for_change = y_train_for_change.iloc[intexes_for_change_no_dublicates]
y_train_for_change['Winner'] = 1.0
y_train_for_change
#changing all these instance's winners from 0 to 1

X_train_changed = pd.concat([X_train_for_change_B, X_train_for_change_R, X_train_for_change_diff], axis=1)
#merge all the changed datasets to a final changed dataset with all the features

X_train_without_pca_full_dropped = X_train_without_pca_full.copy()
X_train_without_pca_full_dropped.drop(intexes_for_change_no_dublicates, inplace=True)
X_train_without_pca_full_dropped.index

X_train_without_pca_full_changed = pd.concat([X_train_changed, X_train_without_pca_full_dropped]) 
X_train_without_pca_full_changed.index

y_train_full_dropped = y_train_full.copy()
y_train_full_dropped.drop(intexes_for_change_no_dublicates, inplace=True)
y_train_full_changed = pd.concat([y_train_for_change, y_train_full_dropped]) 
(y_train_full_changed == 0).sum() / (y_train_full_changed == 1).sum()
#drop the instances we marked from the original dataset and merge them with the same instances changed

ind_list=[i for i in range(len(y_train_full_changed))]
random.shuffle(ind_list)
ind_list
y_train_full_changed = y_train_full_changed.iloc[ind_list].reset_index(drop=True)
X_train_without_pca_full_changed = X_train_without_pca_full_changed.iloc[ind_list].reset_index(drop=True)
#shuffle the instances


pca = PCA(n_components=23)
X_train_pca_full_changed_values = pca.fit_transform(X_train_without_pca_full_changed)
X_train_pca_full_changed_values
pca.explained_variance_ratio_.sum()
#0.95


X_train_pca_full_changed = pd.DataFrame(X_train_pca_full_changed_values, columns=pca.get_feature_names_out(), index=X_train_without_pca_full_changed.index)

X_test_without_pca_full = X_test_without_pca_full[X_train_without_pca_full_changed.columns]
#changing the order of the features to be the same as the fitted pca's dataset
X_test_pca_full_changed_values = pca.transform(X_test_without_pca_full)
X_test_pca_full_changed = pd.DataFrame(X_test_pca_full_changed_values, columns=pca.get_feature_names_out(), index=X_test_without_pca_full.index)
#Doing pca in the training and test set, n components equal to 23 to fit our trained NN 
X_train_pca_full_changed.to_csv('X_train_pca_full_changed.csv', index=False)
X_test_pca_full_changed.to_csv('X_test_pca_full_changed.csv', index=False)
y_train_full_changed.to_csv('y_train_full_changed.csv', index=False)
#Saving the datasets
(y_train_full_changed == 0).sum() / (y_train_full_changed == 1).sum()
'''
#training sets where we changed some values of 0 to 1 based on keeping the rates of instances found in the GaussianMixtures of 10 n_components fitted
X_train_pca_full_changed = pd.read_csv("./X_train_pca_full_changed.csv")
X_test_pca_full_changed = pd.read_csv("./X_test_pca_full_changed.csv")
y_train_full_changed = pd.read_csv("./y_train_full_changed.csv")


weight_minoniry_class_full = ((y_train_full == 0).sum() / (y_train_full == 1).sum()).values[0]



#Using the keras tuner library to search hyperparameters for a NN
def build_model(hp):
    dnn_layers_ss = [1,2,3,4,5,6,7,8,9]
    dnn_units_min, dnn_units_max = 32, 712
    dr_rate_min, dr_rate_max = 0.1, 0.5
    active_func_ss = ['swish', 'gelu']
    optimizer_ss = ['adamW']
    lr_min, lr_max = 1e-4, 1e-1
    l2_min, l2_max = 0, 0.30
    
    active_func = hp.Choice('activation', active_func_ss)
    optimizer = hp.Choice('optimizer', optimizer_ss)
    lr = hp.Float('learning_rate', min_value=lr_min, max_value=lr_max, sampling='log')
    regul = hp.Float('l2', min_value=l2_min, max_value=l2_max)
    inputs = tf.keras.Input(shape=(X_train_pca.shape[1]))
  
    
    # create hidden layers
    dnn_units = hp.Int(f"0_units", min_value=dnn_units_min, max_value=dnn_units_max)
    dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(regul))(inputs)
    tf.keras.layers.BatchNormalization()(dense)
    for layer_i in range(hp.Choice("n_layers", dnn_layers_ss) - 1):
        dnn_units = hp.Int(f"{layer_i}_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(regul))(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        if hp.Boolean("dropout"):
            dr_rate = hp.Float('dr_rate', min_value=dr_rate_min, max_value=dr_rate_max)
            dense = tf.keras.layers.Dropout(rate=dr_rate)(dense)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    if optimizer == "adamW":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise("Not supported optimizer")
        
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['AUC'])
    return model

def build_tuner(model, hpo_method, objective, dir_name):
    if hpo_method == "RandomSearch":
        tuner = kt.RandomSearch(model, objective=objective, max_trials=3, executions_per_trial=1,
                               project_name=hpo_method, directory=dir_name)
    elif hpo_method == "Hyperband":
        tuner = kt.Hyperband(model, objective=objective, max_epochs=20, executions_per_trial=2,
                            project_name=hpo_method, overwrite=True)
    elif hpo_method == "BayesianOptimization":
        tuner = kt.BayesianOptimization(model, objective=objective, max_trials=10, executions_per_trial=1,
                                       project_name=hpo_method)
    return tuner
  
obj = kt.Objective('val_auc', direction='max')
dir_name = "v1"
Hyperband_tuner = build_tuner(build_model, "Hyperband", obj, dir_name)
Hyperband_tuner.search(X_train_pca, y_train_final, epochs=5, validation_data= (X_valid_pca, y_valid_final), class_weight = {0:1, 1:weight_minoniry_class_full})

top3_models = Hyperband_tuner.get_best_models(num_models = 3)
best_mod = top3_models[0]

top3_params = Hyperband_tuner.get_best_hyperparameters(num_trials = 3)
best_hyp = top3_params[0].values
'''
{'activation': 'gelu',
 'optimizer': 'adamW',
 'learning_rate': 0.01540787095021625,
 'l2': 0.08245190881911092,
 '0_units': 398,
 'n_layers': 2,
 'dropout': True,
 '1_units': 418,
 '2_units': 374,
 '3_units': 131,
 '4_units': 508,
 'dr_rate': 0.4158562653630815,
 '5_units': 530,
 '6_units': 327,
 '7_units': 296,
 'tuner/epochs': 20,
 'tuner/initial_epoch': 7,
 'tuner/bracket': 2,
 'tuner/round': 2,
 'tuner/trial_id': '0012'}
'''

second_mod = top3_models[1]
second_hyp = top3_params[1].values
'''
{'activation': 'gelu',
 'optimizer': 'adamW',
 'learning_rate': 0.0008556581060099099,
 'l2': 0.24416588766106204,
 '0_units': 291,
 'n_layers': 6,
 'tuner/epochs': 20,
 'tuner/initial_epoch': 7,
 'tuner/bracket': 2,
 'tuner/round': 2,
 'dropout': False,
 '1_units': 32,
 '2_units': 32,
 '3_units': 32,
 '4_units': 32,
 'tuner/trial_id': '0015',
 'dr_rate': 0.3633049831509424,
 '5_units': 701,
 '6_units': 61,
 '7_units': 248}
'''
third_mod = top3_models[2]
third_hyp = top3_params[2].values


'''
{'activation': 'swish',
 'optimizer': 'adamW',
 'learning_rate': 0.0013481629442763257,
 'l2': 0.11901563594343909,
 '0_units': 482,
 'n_layers': 1,
 'dropout': False,
 '1_units': 262,
 '2_units': 387,
 '3_units': 144,
 '4_units': 530,
 'dr_rate': 0.34711629941809663,
 '5_units': 248,
 '6_units': 563,
 '7_units': 319,
 'tuner/epochs': 7,
 'tuner/initial_epoch': 0,
 'tuner/bracket': 1,
 'tuner/round': 0}
'''

best_mod.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc' ,patience=20, restore_best_weights=True)
third_mod.fit(X_train_pca, y_train_final, epochs= 100, class_weight = {0:1, 1:weight_minoniry_class_full}, callbacks=[callback], validation_data=(X_valid_pca, y_valid_final))
history = third_mod.fit(X_train_pca_full, y_train_full, epochs= 10, class_weight = {0:1, 1:weight_minoniry_class_full})
best_mod.evaluate(x=X_test_pca, y=y_test_final)
top_mod_NN = third_mod

metrics_for_test_set(model=top_mod_NN, test_set_x=X_test_fair_pca, test_set_y=y_test_fair_full, threshold=0.6, samples=1)
'''
for the 3rd_mod
for threshold 0.64
precision for zero is 0.659 and recall for zero is 0.149 
precision for one is 0.712 and recall for one is 0.134
percentage of none is 79.3%
average precision for both classes is 0.686
average recall for both classes is 0.141
'''

# third_mod.save("third_mod_NN.h5")

pd.DataFrame(history.history).plot(figsize=(8, 5))


#It seemed that the model that i tuned and trained earlier outperformed all the 3 best models of the hyperparameter search so i will be using this

'''
model.save("62acc.h5") 62%acc, X with pca and cat
'''
#the problem is that this model has an input of 29 (with the cat features) but all the other models in the final ensemble uses the training set with 23 features,so we will try transfer learning from the other model to a new one with the right amount of inputs
'''
top_mod_NN = tf.keras.models.load_model('./62acc.h5')

model_clone = tf.keras.models.clone_model(top_mod_NN, input_tensors=tf.keras.layers.Input(shape=X_train_pca.shape[1:]))
for i in range(2, len(model_clone.layers)):
    model_clone.layers[i].set_weights(top_mod_NN.layers[i].get_weights())

model_clone.summary()

for layer in model_clone.layers[:-1]:
    layer.trainable = False

model_clone.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history_test = model_clone.fit(X_train_pca, y_train_final, epochs=100, validation_data=(X_valid_pca, y_valid_final), batch_size=32, callbacks=[callback], class_weight= {0:1, 1:weight_minoniry_class})

for layer in model_clone.layers[:-1]:
    layer.trainable = True

model_clone.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history_test = model_clone.fit(X_train_pca, y_train_final, epochs=100, validation_data=(X_valid_pca, y_valid_final), batch_size=32, callbacks=[callback], class_weight= {0:1, 1:weight_minoniry_class})


model_clone.evaluate(x=X_test_pca, y=y_test_final)

model_clone.save("61acc.h5")  61acc, X pca without cat
'''
#the NN model i will be using
top_mod_NN = tf.keras.models.load_model('./61acc.h5')


possible_n_est = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
possible_max_depth = [1, 2, 3, 4, 5, 6, 7]
class_weight = (y_train_full == 0).sum() / (y_train_full == 1).sum()
for n_est in possible_n_est:
    for max_depth in possible_max_depth:
        rnf_clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42, class_weight={0:1 , 1:class_weight})
        rnf_clf.fit(X_train_pca_full, y_train_full)
        train_acc = (rnf_clf.predict(X_train_pca) == y_train_final).sum() / X_train_pca.shape[0]
        test_acc = (rnf_clf.predict(X_test_pca) == y_test_final).sum() / X_test_pca.shape[0]
        print(f"For n_est:{n_est} and max_depth:{max_depth} we have train_acc:{train_acc} and test_acc{test_acc}")
#the best test accuracy is  59.59% for n_est:400 and max_depth:6, so we are gonna narrow our bayesian hyperparameter search around those numbers

#we will rerun the optimization phase now with the same weight class ratios dataset and with the optimization metric to be the roc-auc score
#bayesian hyperparameter search for a RandomForestClassifier
def optimize_RFC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    
    kf = StratifiedKFold(n_splits=5)
    auc_scores = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        
        model = RandomForestClassifier(**params, class_weight={0:1, 1:1.45})
        model.fit(xtrain, ytrain)
        preds = model.predict_proba(xtest)
        winner_one = preds[:, 1]
        auc = roc_auc_score(ytest, winner_one)
        auc_scores.append(auc)
    
    return -1.0 * np.mean(auc_scores)



param_space_RFC = [
    space.Integer(3, 20, name="max_depth"),
    space.Integer(200, 900, name="n_estimators"),
    space.Categorical(["gini", "entropy"], name="criterion"),
    space.Real(0.01, 1, prior="uniform" ,name="max_features"),
    space.Real(0.01, 1, prior="uniform" ,name="max_samples"),
]
param_names_RFC = [
    "max_depth",
    "n_estimators",
    "criterion",
    "max_features",
    "max_samples"
]


optimization_function_RFC = partial(
    optimize_RFC,
    param_names = param_names_RFC,
    x = X_train_pca_full,
    y = y_train_full
)

result_RFC =  gp_minimize(
    optimization_function_RFC,
    dimensions = param_space_RFC,
    verbose=10,
    n_calls=30
)

print(
    dict(zip(param_names_RFC, result_RFC.x))
)
#for the same_weight_class_ratios and balanced dataset
#{'max_depth': 13, 'n_estimators': 673, 'criterion': 'gini', 'max_features': 0.9901292474497067}
#{'max_depth': 15, 'n_estimators': 334, 'criterion': 'entropy', 'max_features': 0.01115001788516646}
#for the same_weight_class_ratios  unbalanced dataset and class_weight={0:1, 1:weight_minoniry_class_full
#max_depth= 3, n_estimators= 700, criterion= entropy, max_features= 0.18396804650705606, max_samples= 0.1
#for the same_weight_class_ratios  unbalanced dataset and class_weight={0:1, 1:1.45}

#final model: top_mod_RFC = RandomForestClassifier(max_depth= 6, n_estimators= 858, criterion= 'gini', max_features= 0.26726799261727396, max_samples= 0.35215557719305063, class_weight={0:1, 1:1.45}, random_state=True)
top_mod_RFC = RandomForestClassifier(max_depth= 6, n_estimators= 858, criterion= 'gini', max_features= 0.26726799261727396, max_samples= 0.35215557719305063, class_weight={0:1, 1:1.45}, random_state=True)



#bayesian hyperparameter search for a GradientBoostingClassifier
def optimize_XGB(params, param_names, x, y):
    params= dict(zip(param_names, params))
    kf = StratifiedKFold(n_splits=5)
    auc_scores = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        
        model = XGBClassifier(**params, scale_pos_weight= 1.3913330873665453)
        model.fit(xtrain, ytrain)
        preds = model.predict_proba(xtest)
        winner_one = preds[:, 1]
        auc = roc_auc_score(ytest, winner_one)
        auc_scores.append(auc)
    return -1.0 * np.mean(auc_scores)

param_space_XGB = [
    space.Integer(2, 8, name="max_depth"),
    space.Real(0.001, 1.0, prior="log-uniform" ,name="learning_rate"),
    space.Real(0.1, 1.0, prior="uniform" ,name="subsample"),
    space.Real(0.3, 1.0, prior="uniform" ,name="colsample_bytree"),
    space.Real(0.3, 1.0, prior="uniform" ,name="colsample_bylevel"),
    space.Real(0.3, 1.0, prior="uniform" ,name="colsample_bynode"),
    space.Real(0.0, 10.0, prior="uniform" ,name="reg_alpha"),
    space.Real(0.0, 10.0, prior="uniform" ,name="reg_lambda"),
    space.Real(0.0, 15.0, prior="uniform" ,name="gamma"),
    
]
param_names_XGB = [
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "colsample_bynode",
    "reg_alpha",
    "reg_lambda",
    "gamma",
   
    
]

optimization_function_XGB = partial(
    optimize_XGB,
    param_names = param_names_XGB,
    x = X_train_pca_full,
    y = y_train_full
)

result_XGB =  gp_minimize(
    optimization_function_XGB,
    dimensions = param_space_XGB,
    verbose=10,
    n_calls=100,
)

print(
    dict(zip(param_names_XGB, result_XGB.x))
)
#for the 'changed' x, y
#max_depth= 8, learning_rate= 0.14853209522354252, subsample= 1.0, colsample_bytree= 1.0, colsample_bylevel= 1.0, colsample_bynode= 0.3, reg_alpha= 0.0, reg_lambda= 10.0, gamma= 3.032654722224418, scale_pos_weight= 0.899401006807208
#for the unbalanced dataset
#max_depth= 2, learning_rate= 0.018649572919629702, subsample= 0.3082460927861192, colsample_bytree= 1.0, colsample_bylevel= 0.8108681922822245, colsample_bynode= 1.0, reg_alpha= 0.0, reg_lambda= 0.0, gamma= 10.0, scale_pos_weight= 1.1196603874123938
#for the unbalanced dataset and fixes scale_pos_weight at the weight minority class full
#max_depth= 2, learning_rate= 0.03310425369858327, subsample= 0.3, colsample_bytree= 1.0, colsample_bylevel= 1.0, colsample_bynode= 1.0, reg_alpha= 0.0, reg_lambda= 0.0, gamma= 0.0
#for the unbalanced dataset and  scale_pos_weight= 1.3913330873665453
#max_depth= 3, learning_rate= 0.025131944380062544, subsample= 0.4349890855990146, colsample_bytree= 1.0, colsample_bylevel= 0.9098375303771553, colsample_bynode= 0.6677042615417232, reg_alpha= 2.0554171330772206, reg_lambda= 4.620076924012009, gamma= 5.651886443117005, scale_pos_weight= 1.3913330873665453

#final model: top_mod_XGB = XGBClassifier(max_depth= 3, learning_rate= 0.025131944380062544, subsample= 0.4349890855990146, colsample_bytree= 1.0, colsample_bylevel= 0.9098375303771553, colsample_bynode= 0.6677042615417232, reg_alpha= 2.0554171330772206, reg_lambda= 4.620076924012009, gamma= 5.651886443117005, scale_pos_weight= 1.3913330873665453)
top_mod_XGB = XGBClassifier(max_depth= 3, learning_rate= 0.025131944380062544, subsample= 0.4349890855990146, colsample_bytree= 1.0, colsample_bylevel= 0.9098375303771553, colsample_bynode= 0.6677042615417232, reg_alpha= 2.0554171330772206, reg_lambda= 4.620076924012009, gamma= 5.651886443117005, scale_pos_weight= 1.3913330873665453)


#bayesian hyperparameter search for a LogisticRegression as a meta learner
#The choice of a LogisticRegression model is that it gives calibrated estimated probabilities as the output and that the more powerful models would overfit for sure.We restrain the C values to be very small for the same reason as well 
def optimize_meta(params, param_names, x, y):
    params= dict(zip(param_names, params))
    kf = StratifiedKFold(n_splits=5)
    auc_scores = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        
        model = LogisticRegression(**params, class_weight = {0:1, 1:1.5})
        model.fit(xtrain, ytrain)
        preds = model.predict_proba(xtest)
        winner_one = preds[:, 1]
        auc = roc_auc_score(ytest, winner_one)
        auc_scores.append(auc)
    return -1.0 * np.mean(auc_scores)

param_space_meta = [
    space.Real(0.01, 10.0, prior="log-uniform" ,name="C"),
    space.Integer(100, 300, name="max_iter"),    
    space.Categorical(["liblinear", "newton-cg", "lbfgs", "newton-cholesky" ], name="solver")
]
param_names_meta = [
    "C",
    "max_iter",
    "solver"
]

optimization_function_meta = partial(
    optimize_meta,
    param_names = param_names_meta,
    x = estimators_training,
    y = y_train_full
)

result_meta =  gp_minimize(
    optimization_function_meta,
    dimensions = param_space_meta,
    verbose=10,
    n_calls=100,
)

print(
    dict(zip(param_names_meta, result_meta.x))
)

#max_iter = 163, solver = 'newton-cholesky', class_weight = {0:1, 1:minority...}
#C= 100.0, max_iter= 300, solver= 'liblinear'

#final model meta_learner = LogisticRegression(C= 10.0, max_iter= 300, solver= 'liblinear', class_weight = {0:1, 1:1.5})
meta_learner = LogisticRegression(C= 10.0, max_iter= 300, solver= 'liblinear', class_weight = {0:1, 1:1.5})




#naive bayes classifier
bayes = GaussianNB()


#So we end up with four classifiers : NN, RFC, XGB, and GaussianNaiveBayes.The fact that the classifiers are adequately independed will provide enough diversity on the type of errors they make.So that the ensemble at the end will provide better predictions




#Our classifiers will predict that the class is one if the probabilities it gives for that class is over the number threshold
#Will predict that the class is zero if the probabilities it gives for the class one is under the number 1 - threshold
#Else it will not give a prediction of the class (appends it as -1)
#We configure the classifiers that way because it enables to pick a threshold that gives high accuracy for both classes

#This function is created to be passed on the fit method of the GaussianNB() and it will give a weight of 1 to the zero class and a weight of (weight_minoniry_class_full) to the the one class during training
def class_weight(y):
    weights = []
    for i in range(len(y)):
        if y.values[i][0] == 0:
            weights.append(1)
        if y.values[i][0] == 1:
            weights.append(weight_minoniry_class_full)
    return weights
            

#This function will return a list of predictions by the model on the test_set_x
def model_gen_test(model, test_set, threshold, samples=None):
    if model == top_mod_NN:
        if samples == 1:
            win_one = model.predict(test_set)
        if samples > 1:
        #monte carlo method
            preds = np.stack([model(test_set, training=True) for sample in range(samples)])
            win_one = preds.mean(axis=0)

    else:
         y_probas = model.predict_proba(test_set)
         win_one = y_probas[:, 1]
         
    thres_preds = []
    for i in range(len(test_set)):
        if win_one[i] >= (threshold):
            thres_preds.append(1)
        if win_one[i] <= (1 - threshold):
            thres_preds.append(0)
        elif ((win_one[i] < (threshold)) and (win_one[i] > (1 - threshold))):
            thres_preds.append(-1)
    return(thres_preds)



#This function will return the classification metrics for both classes (plus a percentage of nonclassified instances)
def metrics_for_both_classes(model, test_set_x, test_set_y, threshold, samples=None):
    preds_test = model_gen_test(model = model, test_set = test_set_x, threshold = threshold, samples=samples)
    predicted_zero = 0
    predicted_one = 0
    predicted_none = 0
    correct_zero = 0
    correct_one = 0
    winner_zero = (test_set_y == 0).reset_index()['Winner']
    winner_one = (test_set_y == 1).reset_index()['Winner']
    for i in range(len(test_set_x)):
        if (preds_test[i] == 0):
            predicted_zero = predicted_zero + 1
            if winner_zero[i]:
                correct_zero = correct_zero + 1
        if (preds_test[i] == 1):
            predicted_one = predicted_one + 1
            if winner_one[i]:
                correct_one = correct_one + 1
        if (preds_test[i] == -1):
            predicted_none = predicted_none + 1
    if predicted_one == 0:
        predicted_one = 1
    if predicted_zero == 0:
        predicted_zero = 1
    #avoid division by zero
    precision_zero = correct_zero / predicted_zero
    recall_zero = correct_zero / winner_zero.sum()
    precision_one = correct_one / predicted_one
    recall_one = correct_one / winner_one.sum()
    percentage_of_none = predicted_none / len(test_set_y)
    return precision_zero, precision_one, recall_zero, recall_one, percentage_of_none
    



#this function will print the average classification scores of a model on 10 different test sets (who have equal number of negative and positive classes).The model is fitted on a different training set (would be x - test set) that 10 times 
def probas_stratify_kfolds(model, x, y, threshold, samples=None):
    model = model
    kf = StratifiedKFold(n_splits=10)
    recalls_one = []
    precisions_one = []
    recalls_zero = []
    precisions_zero = []
    avg_precisions = []
    avg_recalls = []
    avg_nones = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        if model != top_mod_NN:
            model.fit(xtrain, ytrain)
        if model == bayes:
            model.fit(xtrain, ytrain, sample_weight=class_weight(xtrain, ytrain))
        precision_zero, precision_one,  recall_zero, recall_one, percentage_of_none  = metrics_for_both_classes(model = model, test_set_x = xtest, test_set_y= ytest, threshold = threshold, samples=samples)
        avg_precision = (precision_one + precision_zero) / 2
        avg_recall = (recall_one + recall_zero) / 2
        precisions_one.append(precision_one)
        recalls_one.append(recall_one)
        precisions_zero.append(precision_zero)
        recalls_zero.append(recall_zero)        
        avg_precisions.append(avg_precision)
        avg_recalls.append(avg_recall)
        avg_nones.append(percentage_of_none)
        
        
    avg_prec_both = sum(avg_precisions) / len(avg_precisions)
    avg_rec_both = sum(avg_recalls) / len(avg_recalls)
    avg_prec_one = sum(precisions_one) / len(precisions_one)
    avg_prec_zero = sum(precisions_zero) / len(precisions_zero)
    avg_rec_one = sum(recalls_one) / len(recalls_one)
    avg_rec_zero = sum(recalls_zero) / len(recalls_zero)
    avg_none = sum(avg_nones) / len(avg_nones)
    if model == top_mod_NN:
        return print(f"for threshold {threshold} and samples {samples}\nprecision for zero is {round(avg_prec_zero, 3)} and recall for zero is {round(avg_rec_zero, 3)} \nprecision for one is {round(avg_prec_one, 3)} and recall for one is {round(avg_rec_one, 3)}\npercentage of none is {100 * round(avg_none, 3)}%\naverage precision for both classes is {round(avg_prec_both, 3)}\naverage recall for both classes is {round(avg_rec_both, 3)}\n\n")
    else:
        return print(f"for threshold {threshold}\nprecision for zero is {round(avg_prec_zero, 3)} and recall for zero is {round(avg_rec_zero, 3)} \nprecision for one is {round(avg_prec_one, 3)} and recall for one is {round(avg_rec_one, 3)}\npercentage of none is {100 * round(avg_none, 3)}%\naverage precision for both classes is {round(avg_prec_both, 3)}\naverage recall for both classes is {round(avg_rec_both, 3)}\n\n")


#Its time to choose the right balance between precision and recall.I believe that the metric that i should give the most of the weight is precision (the average precision of the two classes)
#I look at it as a gambler's view.Its way more important that you are more certain about the correct winner than finding the most correct winners, if you are gonna bet for the winner.
#we will look for over 10% recall.That means for every ufc card (about 15 fights) it will surely give a confident answer
#On top of that its very important to have similar values of recall for both classes for one threshold.That will be a good indicator that our model is not highly bias to pick one class over the other

#The way to adjust the precision-recall balance is via the threshold value (and the samples for the monte carlo method on the NN).It only makes sense to take values over 0.5 (because the prediction for zero is when this value is under 1 - threshold)
for i in range(50, 57, 1):
    probas_stratify_kfolds(top_mod_RFC, x=X_train_pca_full, y=y_train_full, threshold=i / 100)

'''
for same_weight_class_ratio and X_train_pca_full_changed
for threshold 0.58
precision for zero is 0.544 and recall for zero is 0.126 
precision for one is 0.597 and recall for one is 0.114
percentage of none is 78.5%
average precision for both classes is 0.57
average recall for both classes is 0.12
'''
'''
for same_weight_class_ratio and X_train_pca_full and class weights equal to minority...
for threshold 0.53
precision for zero is 0.71 and recall for zero is 0.339 
precision for one is 0.611 and recall for one is 0.207
percentage of none is 58.199999999999996%
average precision for both classes is 0.66
average recall for both classes is 0.273
'''
'''
for the same_weight_class_ratios  unbalanced dataset and class_weight={0:1, 1:1.45}
for threshold 0.56
precision for zero is 0.719 and recall for zero is 0.306 
precision for one is 0.585 and recall for one is 0.281
percentage of none is 55.1%
average precision for both classes is 0.652
average recall for both classes is 0.294
'''

for i in range(50, 65, 1):
    probas_stratify_kfolds(top_mod_XGB, x=X_train_pca_full, y=y_train_full, threshold = i / 100)
'''
for same_weight_class_ratio and X_train_pca_full_changed
for threshold 0.65
precision for zero is 0.548 and recall for zero is 0.199 
precision for one is 0.573 and recall for one is 0.115
percentage of none is 71.7%
average precision for both classes is 0.56
average recall for both classes is 0.157
'''
'''
for same_weight_class_ratio and X_train_pca_full
for threshold 0.54
precision for zero is 0.661 and recall for zero is 0.611 
precision for one is 0.618 and recall for one is 0.137
percentage of none is 37.1%
average precision for both classes is 0.639
average recall for both classes is 0.374
'''
'''
for same_weight_class_ratio and X_train_pca_full and fixed scale_pos_weight 
for threshold 0.62
precision for zero is 0.747 and recall for zero is 0.142 
precision for one is 0.612 and recall for one is 0.109
percentage of none is 81.5%
average precision for both classes is 0.68
average recall for both classes is 0.125
'''
'''
for same_weight_class_ratio and X_train_pca_full and scale_pos_weight =  1.39
for threshold 0.59
precision for zero is 0.731 and recall for zero is 0.2 
precision for one is 0.605 and recall for one is 0.196
percentage of none is 70.5%
average precision for both classes is 0.668
average recall for both classes is 0.198
'''

for i in range(65, 75, 1):
    probas_stratify_kfolds(bayes, x=X_train_pca_full, y=y_train_full, threshold = i / 100)

'''
for threshold 0.7
precision for zero is 0.743 and recall for zero is 0.164 
precision for one is 0.611 and recall for one is 0.11
percentage of none is 79.60000000000001%
average precision for both classes is 0.677
average recall for both classes is 0.137
'''
for i in range(50, 70, 1):
    probas_stratify_kfolds(meta_learner, x=estimators_training, y=y_train_full, threshold = i/100)



for i in range(62, 63, 1):
    for j in range(1, 3, 1):
        probas_stratify_kfolds(top_mod_NN, x=X_train_pca_full, y=y_train_full, threshold = i / 100, samples=j)

'''
for threshold 0.61 and samples 1
precision for zero is 0.811 and recall for zero is 0.16 
precision for one is 0.671 and recall for one is 0.447
percentage of none is 60.6%
average precision for both classes is 0.741
average recall for both classes is 0.303
'''

#fitting all the classifiers on the full training set
top_mod_RFC.fit(X_train_pca_full, y_train_full)
top_mod_XGB.fit(X_train_pca_full, y_train_full)
bayes.fit(X_train_pca_full, y_train_full, sample_weight=class_weight(y_train_full))

#will return a print of the metrics of a model on a test set
def metrics_for_test_set(model, test_set_x, test_set_y, threshold, samples=None):
    precision_zero, precision_one, recall_zero, recall_one, percentage_of_none = metrics_for_both_classes(model = model, test_set_x = test_set_x, test_set_y = test_set_y, threshold= threshold, samples=samples)
    avg_prec = (precision_zero + precision_one) / 2
    avg_rec = (recall_one + recall_zero) / 2
    return print(f"for threshold {threshold}\nprecision for zero is {round(precision_zero, 3)} and recall for zero is {round(recall_zero, 3)} \nprecision for one is {round(precision_one, 3)} and recall for one is {round(recall_one, 3)}\npercentage of none is {100 * round(percentage_of_none, 3)}%\naverage precision for both classes is {round(avg_prec, 3)}\naverage recall for both classes is {round(avg_rec, 3)}\n\n")

#will return a pandas dataframe that for every row of (test_set_x) we have four columns.Each column has the value of the predicted probability (divided by 4) of the instance (of the test_set_x) beloning to class 1, for the four classifiers.
def aver_pred_proba(test_set_x):
    XGB_pred = top_mod_XGB.predict_proba(test_set_x)
    RFC_pred = top_mod_RFC.predict_proba(test_set_x)
    bayes_pred = bayes.predict_proba(test_set_x)
    NN_pred_win_one_ = top_mod_NN.predict(test_set_x)
    XGB_pred_win_one = pd.Series(XGB_pred[:, 1]).apply(lambda x: x/4).tolist()
    RFC_pred_win_one = pd.Series(RFC_pred[:, 1]).apply(lambda x: x/4).tolist()
    bayes_pred_win_one = pd.Series(bayes_pred[:, 1]).apply(lambda x: x/4).tolist()
    NN_pred_win_one = (NN_pred_win_one_ / 4)[:,0].tolist()
    df = pd.DataFrame(data = {'XGB_pred': XGB_pred_win_one, 'RFC_pred': RFC_pred_win_one, 'bayes_pred': bayes_pred_win_one, 'NN_pred': NN_pred_win_one})
    return df

#We will use the predicted probabilities of the four classifiers on the training set as the features of a meta learner classifier
 
estimators_training = aver_pred_proba(test_set_x=X_train_pca_full)
meta_learner.fit(estimators_training, y_train_full)
meta_learner.coef_
estimators_test = aver_pred_proba(test_set_x=X_test_fair_pca)


for i in range(50, 99, 1):
    metrics_for_test_set(model=meta_learner, test_set_x=estimators_test, test_set_y=y_test_fair_full, threshold=i / 100)

'''
for the meta_learner as a logistic regression model with max_iter = 163, solver = 'newton-cholesky', class_weight = {0:1, 1:weight_minoniry_class_full}
NN the 61acc
for threshold 0.67
precision for zero is 0.607 and recall for zero is 0.139 
precision for one is 0.725 and recall for one is 0.149
percentage of none is 78.3%
average precision for both classes is 0.666
average recall for both classes is 0.144
'''
'''
for the meta_learner as a logistic regression model with max_iter = 163, solver = 'newton-cholesky', class_weight = {0:1, 1:weight_minoniry_class_full}
NN the third
for threshold 0.64
precision for zero is 0.632 and recall for zero is 0.203 
precision for one is 0.718 and recall for one is 0.131
percentage of none is 74.8%
average precision for both classes is 0.675
average recall for both classes is 0.167
'''

'''
meta_learner = LogisticRegression(C= 10.0, max_iter= 300, solver= 'liblinear', class_weight = {0:1, 1:1.5})
NN the 61acc
for threshold 0.77
precision for zero is 0.69 and recall for zero is 0.126 
precision for one is 0.743 and recall for one is 0.134
percentage of none is 81.89999999999999%
average precision for both classes is 0.716
average recall for both classes is 0.13
'''




