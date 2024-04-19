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
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier

dataset_choice = "same_y_ratios" #Choose between "same_weight_class_ratios" or "same_y_ratios"

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

        for index in y_test_fair_full.index:
            ratio = ((y_test_fair_full == 1).sum() / (y_test_fair_full == 0).sum())[0]
            if (ratio < 1):
                if (y_test_fair_full.loc[index].values[0] == 0):
                    y_test_fair_full.drop(index, inplace=True)
                    X_test_fair_full.drop(index, inplace=True)
            
        y_test_full = y_test_same_weight_class_ratios.copy()
        X_test_full = X_test_full_same_weight_class_ratios.copy()
        y_train_full = y_train_same_weight_class_ratios.copy()
        X_train_full = X_train_full_same_weight_class_ratios.copy()

        X_train_without_pca_full = X_train_without_pca_full_same_weight_class_ratios.copy()  
        X_test_without_pca_full = X_test_without_pca_full_same_weight_class_ratios.copy()
    
    if dataset_ch == "same_y_ratios":
        X_train_full_same_y_ratios = pd.read_csv('./X_train_concat_branch.csv', dtype=np.float64)
        X_test_full_same_y_ratios= pd.read_csv('./X_test_concat_branch.csv', dtype=np.float64)
        y_train_same_y_ratios= pd.read_csv('./y_train_branch.csv', dtype=np.float64)
        y_test_same_y_ratios= pd.read_csv('./y_test_branch.csv', dtype=np.float64)

        X_train_without_pca_full_same_y_ratios= pd.read_csv('./X_train_num_scaled_branch.csv', dtype=np.float64)
        X_test_without_pca_full_same_y_ratios= pd.read_csv('./X_test_num_scaled_branch.csv', dtype=np.float64)
        
        y_test_fair_full = y_test_same_y_ratios.copy()
        X_test_fair_full = X_test_full_same_y_ratios.copy()

        for index in y_test_fair_full.index:
            ratio = ((y_test_fair_full == 1).sum() / (y_test_fair_full == 0).sum())[0]
            if (ratio < 1):
                if (y_test_fair_full.loc[index].values[0] == 0):
                    y_test_fair_full.drop(index, inplace=True)
                    X_test_fair_full.drop(index, inplace=True)
        y_test_full = y_test_same_y_ratios.copy()
        X_test_full = X_test_full_same_y_ratios.copy()
        y_train_full = y_train_same_y_ratios.copy()
        X_train_full = X_train_full_same_y_ratios.copy()

        X_train_without_pca_full = X_train_without_pca_full_same_y_ratios.copy()
        X_test_without_pca_full = X_test_without_pca_full_same_y_ratios.copy()
    return y_test_full, X_test_full, y_train_full, X_train_full, X_train_without_pca_full, X_test_without_pca_full, X_test_fair_full, y_test_fair_full


y_test_full, X_test_full, y_train_full, X_train_full, X_train_without_pca_full, X_test_without_pca_full, X_test_fair_full, y_test_fair_full = training_test_sets(dataset_choice)


#X with pca and cat
X_train_pca_cat = X_train_full[:3500].select_dtypes(np.float64)
X_valid_pca_cat = X_train_full[3500:].select_dtypes(np.float64)
X_test_pca_cat = X_test_full.copy()

#X with pca without cat
X_train_pca_full = X_train_full.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_train_pca = X_train_pca_cat[:3500].drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_valid_pca = X_valid_pca_cat.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
X_test_pca = X_test_pca_cat.drop(['B_Stance_Orthodox', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Orthodox', 'R_Stance_Southpaw', 'R_Stance_Switch'], axis=1)
#X without pca with cat

#X without pca without cat
X_train_without_pca = X_train_without_pca_full[:3500].select_dtypes(np.float64)
X_valid_without_pca = X_train_without_pca_full[3500:].select_dtypes(np.float64)
X_test_without_pca = X_test_without_pca_full.copy().select_dtypes(np.float64)
#
y_train_final = y_train_full[:3500].select_dtypes(np.float64)['Winner']
y_valid_final = y_train_full[3500:].select_dtypes(np.float64)['Winner']
y_test_final = y_test_full.copy()['Winner']



(y_train_full == 1).sum() / (y_train_full == 0).sum()
(y_test_full == 1).sum() / (y_test_full == 0).sum()

weight_minoniry_class_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()






n_units = 300
activation = tf.keras.activations.gelu
initializer = tf.keras.initializers.he_normal()


model_NN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train_pca.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])




lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1 , decay_steps=10000, decay_rate=0.96 )
optimizer = tf.keras.optimizers.legacy.Nadam(clipnorm=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy' ,patience=20, restore_best_weights=True)

model_NN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#we should put a higher weight in the underpressented class (1 in this case)
weight_minoniry_class = (y_train_final == 0).sum() / (y_train_final == 1).sum()

history = model_NN.fit(X_train_pca, y_train_final, epochs=100, validation_data=(X_valid_pca, y_valid_final), batch_size=32, callbacks=[callback], class_weight= {0:1, 1:weight_minoniry_class})

model.evaluate(x=X_test_pca, y=y_test_final)

pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot(figsize=(8, 5))




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
#the best test accuracy is  59.59% for n_est:400 and max_depth:6, so we are gonna narrow our bayesian hyperparameter search around that number

#we will rerun the optimization phase now with the same y ratios dataset and with the optimization metric to be the mean of the precisions of 0 and 1 classes

def optimize_RFC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    
    kf = StratifiedKFold(n_splits=5)
    presicions = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        weight_minoniry = (ytrain == 0).sum() / (ytrain == 1).sum()
        model = RandomForestClassifier(**params, class_weight={0:1 , 1:weight_minoniry})
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        predicted_zero = 0
        predicted_one = 0
        predicted_none = 0
        correct_zero = 0
        correct_one = 0
        winner_zero = (ytest == 0).reset_index()['Winner']
        winner_one = (ytest == 1).reset_index()['Winner']
        for i in range(len(xtest)):
            if (preds[i] == 0):
                predicted_zero = predicted_zero + 1
                if winner_zero[i]:
                    correct_zero = correct_zero + 1
            if (preds[i] == 1):
                predicted_one = predicted_one + 1
                if winner_one[i]:
                    correct_one = correct_one + 1
        precision_zero = correct_zero / predicted_zero
        precision_one = correct_one / predicted_one
        avg_prec = (precision_one + precision_zero) / 2
        presicions.append(avg_prec)
    
    return -1.0 * np.mean(presicions)



param_space_RFC = [
    space.Integer(3, 15, name="max_depth"),
    space.Integer(200, 700, name="n_estimators"),
    space.Categorical(["gini", "entropy"], name="criterion"),
    space.Real(0.01, 1, prior="uniform" ,name="max_features")
]
param_names_RFC = [
    "max_depth",
    "n_estimators",
    "criterion",
    "max_features"
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
    verbose=10 
)

print(
    dict(zip(param_names_RFC, result_RFC.x))
)
#for the same_y_ratios dataset
#{'max_depth': 15, 'n_estimators': 334, 'criterion': 'entropy', 'max_features': 0.01115001788516646}



top_mod_RFC = RandomForestClassifier(n_estimators=334, max_depth=15, random_state=42, class_weight={0:1 , 1:weight_minoniry_class_full}, criterion='entropy', max_features=0.01115001788516646)


#Its time to choose the right balance between precision and recall.I believe that the metric that i should give the most of the weight is precision.
#I look at it as a gambler's view.Its way more important that you are more certain about the correct winner than finding the most correct winners, if you are gonna bet for the winner.
#we will look for over 10% recall.That means for every ufc card (about 15 fights) it will surely give us one positive class









def optimize_GBC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    kf = StratifiedKFold(n_splits=5)
    presicions = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        
        model = GradientBoostingClassifier(**params, n_estimators=500, n_iter_no_change=10)
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        predicted_zero = 0
        predicted_one = 0
        predicted_none = 0
        correct_zero = 0
        correct_one = 0
        winner_zero = (ytest == 0).reset_index()['Winner']
        winner_one = (ytest == 1).reset_index()['Winner']
        for i in range(len(xtest)):
            if (preds[i] == 0):
                predicted_zero = predicted_zero + 1
                if winner_zero[i]:
                    correct_zero = correct_zero + 1
            if (preds[i] == 1):
                predicted_one = predicted_one + 1
                if winner_one[i]:
                    correct_one = correct_one + 1
        if predicted_zero == 0:
            predicted_zero = 1
        if predicted_one == 0:
            predicted_one = 1
        #avoid division by zero
            
        precision_zero = correct_zero / predicted_zero
        precision_one = correct_one / predicted_one
        avg_prec = (precision_one + precision_zero) / 2
        presicions.append(avg_prec)
    return -1.0 * np.mean(presicions)

param_space_GBC = [
    space.Integer(2, 10, name="max_depth"),
    space.Real(0.8, 1, prior="uniform" ,name="subsample"),
    space.Real(0.01, 1, prior="uniform" ,name="learning_rate")
]
param_names_GBC = [
    "max_depth",
    "subsample",
    "learning_rate"
]

optimization_function_GBC = partial(
    optimize_GBC,
    param_names = param_names_GBC,
    x = X_train_pca_full,
    y = y_train_full
)

result_GBC =  gp_minimize(
    optimization_function_GBC,
    dimensions = param_space_GBC,
    verbose=10 
)

print(
    dict(zip(param_names_GBC, result_GBC.x))
)
#for the same_y_ratios dataset
#{'max_depth': 2, 'subsample': 0.8, 'learning_rate': 0.01}

top_mod_GBC = GradientBoostingClassifier(n_estimators=500, n_iter_no_change=10, max_depth=2, subsample=0.8, learning_rate=0.01)

def optimize_SVC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    model = BaggingClassifier(SVC(kernel='rbf', **params), n_estimators=100, max_features=0.3)
    kf = StratifiedKFold(n_splits=5)
    presicions = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_pres = precision_score(ytest, preds)
        presicions.append(fold_pres)
    return -1.0 * np.mean(presicions)

param_space_SVC = [
    space.Real(0.01, 300, prior="uniform" ,name="C"),
    space.Real(0.1, 2, prior="uniform" ,name="gamma")
]
param_names_SVC = [
    "C",
    "gamma",
]

optimization_function_SVC = partial(
    optimize_SVC,
    param_names = param_names_SVC,
    x = X_train_pca_full,
    y = y_train_full
)

result_SVC =  gp_minimize(
    optimization_function_SVC,
    dimensions = param_space_SVC,
    verbose=10,
    n_calls=30
)

print(
    dict(zip(param_names_SVC, result_SVC.x))
)
#{'C': 1000.0, 'gamma': 0.38279667625542757}
top_mod_SVC = BaggingClassifier(SVC(kernel='rbf', probability=True, class_weight={0:1 , 1:weight_minoniry_class}, C=256.6619199677021, gamma=0.11674269193229327), n_estimators=500, max_samples=100)


def optimize_ABC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), **params)
    kf = StratifiedKFold(n_splits=5)
    presicions = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        predicted_zero = 0
        predicted_one = 0
        predicted_none = 0
        correct_zero = 0
        correct_one = 0
        winner_zero = (ytest == 0).reset_index()['Winner']
        winner_one = (ytest == 1).reset_index()['Winner']
        for i in range(len(xtest)):
            if (preds[i] == 0):
                predicted_zero = predicted_zero + 1
                if winner_zero[i]:
                    correct_zero = correct_zero + 1
            if (preds[i] == 1):
                predicted_one = predicted_one + 1
                if winner_one[i]:
                    correct_one = correct_one + 1
        if predicted_zero == 0:
            predicted_zero = 1
        if predicted_one == 0:
            predicted_one = 1
        precision_zero = correct_zero / predicted_zero
        precision_one = correct_one / predicted_one
        avg_prec = (precision_one + precision_zero) / 2
        presicions.append(avg_prec)
    return -1.0 * np.mean(presicions)

param_space_ABC = [
    space.Integer(20, 40, name="n_estimators"),
    space.Real(0.01, 0.8, prior="uniform" ,name="learning_rate")
]
param_names_ABC = [
    "n_estimators",
    "learning_rate",
]

optimization_function_ABC = partial(
    optimize_ABC,
    param_names = param_names_ABC,
    x = X_train_pca_full,
    y = y_train_full
)

result_ABC =  gp_minimize(
    optimization_function_ABC,
    dimensions = param_space_ABC,
    verbose=10 
)

print(
    dict(zip(param_names_ABC, result_ABC.x))
)
#{'n_estimators': 34, 'learning_rate': 0.10781322577235282}
top_mod_ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=34, learning_rate=0.10781322577235282)

estimators = [
    ('GBC', top_mod_GBC),
    ('ABC', top_mod_ABC),
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')


def model_gen_test(model, test_set, threshold, samples=None):
    if model == top_mod_NN:
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
    



#this function will print the average classification scores of a model on 3 different test sets (who have equal number of negative and positive classes).The model is fitted on a different training set (would be x - test set) that 3 times 
def probas_stratify_kfolds(model, x, y, threshold, samples=None):
    model = model
    kf = StratifiedKFold(n_splits=3)
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
    return print(f"for threshold {threshold}\nprecision for zero is {round(avg_prec_zero, 3)} and recall for zero is {round(avg_rec_zero, 3)} \nprecision for one is {round(avg_prec_one, 3)} and recall for one is {round(avg_rec_one, 3)}\npercentage of none is {100 * round(avg_none, 3)}%\naverage precision for both classes is {round(avg_prec_both, 3)}\naverage recall for both classes is {round(avg_rec_both, 3)}\n\n")


probas_stratify_kfolds(top_mod_RFC, x=X_train_pca_full, y=y_train_full, threshold=0.5)
for i in range(50, 60, 1):
    probas_stratify_kfolds(top_mod_NN, x=X_train_pca_full, y=y_train_full, threshold=i / 100 ,samples=2)

probas_stratify_kfolds(top_mod_NN, x=X_train_pca_full, y=y_train_full, threshold= 0.65 ,samples=2)



def model_gen_test_for_hard_voter(model, test_set, threshold, samples=None):
    if model == top_mod_NN:
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
            thres_preds.append(-1)
        elif ((win_one[i] < (threshold)) and (win_one[i] > (1 - threshold))):
            thres_preds.append(0)
    return(thres_preds)

def hard_voter(test_set):
    RFC_preds = model_gen_test_for_hard_voter(model=top_mod_RFC, test_set=test_set, threshold=50 )
    voting_preds = model_gen_test_for_hard_voter(model=voting_clf, test_set=test_set, threshold=50 )
    NN_preds = model_gen_test_for_hard_voter(model=top_mod_NN, test_set=test_set, threshold=50, samples=2 ) 
    final_preds = []
    for i in range(len(test_set)):
        j = RFC_preds[i] + voting_preds[i] + NN_preds[i]
        if j > 1.5:
            final_preds.append(1)
        if j < (-1.5):
            final_preds.append(0)
        elif ((j < 1.5) and (j > (-1.5))):
            final_preds.append(-1)
    return final_preds



def metrics_for_both_classes_hard_voter(test_set_x, test_set_y):
    preds_test = hard_voter(test_set=test_set_x)
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
    avg_precision = (precision_one + precision_zero) / 2
    avg_recall = (recall_one + recall_zero) / 2
    return print(f"precision for zero is {round(precision_zero, 3)} and recall for zero is {round(recall_zero, 3)} \nprecision for one is {round(precision_one, 3)} and recall for one is {round(recall_one, 3)}\npercentage of none is {100 * round(percentage_of_none, 3)}%\naverage precision for both classes is {round(avg_precision, 3)}\naverage recall for both classes is {round(avg_recall, 3)}\n\n")

top_mod_RFC.fit(X_train_pca_full, y_train_full)
voting_clf.fit(X_train_pca_full, y_train_full)
metrics_for_both_classes_hard_voter(test_set_x=X_test_full, test_set_y=y_test_full)




