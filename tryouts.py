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

X_train_full = pd.read_csv('./X_train_concat.csv', dtype=np.float64)
X_test_full = pd.read_csv('./X_test_concat.csv', dtype=np.float64)
y_train = pd.read_csv('./y_train.csv', dtype=np.float64)
y_test = pd.read_csv('./y_test.csv', dtype=np.float64)

X_train_without_pca_full = pd.read_csv('./X_train_num_scaled.csv', dtype=np.float64)
X_test_without_pca_full = pd.read_csv('./X_test_num_scaled.csv', dtype=np.float64)

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
y_train_final = y_train[:3500].select_dtypes(np.float64)['Winner']
y_valid_final = y_train[3500:].select_dtypes(np.float64)['Winner']
y_test_final = y_test.copy()['Winner']
y_train_full = y_train.copy()


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



def optimize_RFC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    model = RandomForestClassifier(**params)
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
#{'max_depth': 3, 'n_estimators': 700, 'criterion': 'entropy', 'max_features': 0.2264160379499984}



top_mod_RFC = RandomForestClassifier(n_estimators=700, max_depth=3, random_state=42, class_weight={0:1 , 1:weight_minoniry_class}, criterion='entropy', max_features=0.2264160379499984)
top_mod_RFC.fit(X_train_pca, y_train_final)

#Its time to choose the right balance between precision and recall.I believe that the metric that i should give the most of the weight is precision.
#I look at it as a gambler's view.Its way more important that you are more certain about the correct winner than finding the most correct winners, if you are gonna bet for the winner.
#we will look for over 10% recall.That means for every ufc card (about 15 fights) it will surely give us one positive class


RFC_test = RandomForestClassifier(n_estimators=700, max_depth=3, random_state=42, class_weight={0:1 , 1:weight_minoniry_class}, criterion='entropy', max_features=0.2264160379499984)
RFC_test.fit(X_train_pca, y_train_final)
test_pred = RFC_test.predict(X_test_pca)
precision_score(y_test_final, test_pred)
recall_score(y_test_final, test_pred)
f1_score(y_test_final, test_pred)

y_probas = top_mod_RFC.predict_proba(X_test_pca)
y_probas[:, 1][0]

def predict_thres_RFC(threshold):
    thres_preds = []
    for i in range(len(y_probas[:, 1])):
        if y_probas[:, 1][i] > threshold:
            thres_preds.append(1)
        if y_probas[:, 1][i] < threshold:
            thres_preds.append(0)
    return(thres_preds)

len(predict_thres_RFC(0.5))

def RFC_test_scores(threshold):
    preds = predict_thres_RFC(threshold)
    return print(f"for{threshold} threshold we have: precision{precision_score (y_test_final, preds)},recall {recall_score(y_test_final, preds)},f1 {f1_score(y_test_final, preds)}")

for i in range(30, 80, 1):
    RFC_test_scores(i / 100)

'''
for0.54 threshold we have: precision0.5081081081081081,recall 0.2416452442159383,f1 0.32752613240418116
for0.55 threshold we have: precision0.5897435897435898,recall 0.17737789203084833,f1 0.27272727272727276
for0.56 threshold we have: precision0.6086956521739131,recall 0.10796915167095116,f1 0.18340611353711792
for0.57 threshold we have: precision0.675,recall 0.06940874035989718,f1 0.1258741258741259
we keep 0.55 thres
'''




gbcl = GradientBoostingClassifier(max_depth=6, learning_rate=0.05, n_estimators=500, n_iter_no_change=10,random_state=42)

def optimize_GBC(params, param_names, x, y):
    params= dict(zip(param_names, params))
    model = GradientBoostingClassifier(**params, n_estimators=500, n_iter_no_change=10)
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
        fold_pres = precision_score(ytest, preds)
        presicions.append(fold_pres)
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
voting_clf.fit(X_train_pca, y_train_final)
voting_clf.estimators_
predictions_vot_clf = voting_clf.predict(X_test_pca)
for name, clf in voting_clf.named_estimators_.items():
    print(name, "= prec", precision_score(y_test_final, clf.predict(X_test_pca)), ", recall", recall_score(y_test_final, clf.predict(X_test_pca)), ", f1_score", f1_score(y_test_final, clf.predict(X_test_pca)))
#we see that the SVC class is a very weak learner and we should replace it with something better

    
(predictions_vot_clf == y_test_final).sum() / len(y_test_final)
pres_test = precision_score(y_test_final, predictions_vot_clf)
recall_test = recall_score(y_test_final, predictions_vot_clf)
f1_test = f1_score(y_test_final, predictions_vot_clf)



print(f"The voting clf has default precision score of {pres_test} recall score of{recall_test} and f1 score of {f1_test}")
probas = voting_clf.predict_proba(X_test_pca)

win_one = probas[:, 1]

def voting_clf_preds(threshold):
    predictions = []
    for i in range(len(X_test_pca)):
        if win_one[i] > (threshold):
            predictions.append(1)
        if win_one[i] < (threshold):
            predictions.append(0)
    return predictions

voting_scores(0.5)

def voting_scores(threshold):
    preds = voting_clf_preds(threshold=threshold)
    return print(f"for{threshold} threshold we have: precision{precision_score (y_test_final, preds)},recall {recall_score(y_test_final, preds)},f1 {f1_score(y_test_final, preds)}")
    
for i in range(30, 80, 1):
    voting_scores(i / 100)

#for0.51 threshold we have: precision0.6082474226804123,recall 0.15167095115681234,f1 0.24279835390946503

voting_clf_preds(0.51)
j = 0
for i in range(X_test_pca.shape[0]):
    if ((predictions) == 1)[i]:
        if (y_test_final == 1).tolist()[i]:
            j = j + 1
j
((predictions) == 1).sum()
((predictions) == 0).sum()
j / ((predictions) == 1).sum()

NN_preds = top_mod_NN.predict(X_test_pca)
NN_dum = []
for i in NN_preds:
    if i < 0.73:
        NN_dum.append(0)
    if i > 0.73:
        NN_dum.append(1)
NN_dum
precision_score(y_test_final, NN_dum)
recall_score(y_test_final, NN_dum)
f1_score(y_test_final, NN_dum)

#since our NN uses dropout we will try the model carlo dropout trick to have more accurate predictions
def NN_monte_carlo_pred(samples, threshold):
    y_probas = np.stack([top_mod_NN(X_test_pca, training=True) for sample in range(samples)])
    y_proba = y_probas.mean(axis=0)
    monte_carlo_pred = []
    for i in y_proba:
        if i > (threshold):
            monte_carlo_pred.append(1)
        if i < (threshold):
            monte_carlo_pred.append(0)
    return monte_carlo_pred

NN_monte_carlo_pred(2, 0.5)

def monte_carlo_scores(samples, threshold):
    preds = NN_monte_carlo_pred(samples, threshold)
    return print(f"for {samples} samples and {threshold} threshold we have: precision{precision_score (y_test_final, preds)},recall {recall_score(y_test_final, preds)},f1 {f1_score(y_test_final, preds)}")

monte_carlo_scores(1000, 0.5)

for i in (range(2, 100)):
    monte_carlo_scores(i, 0.5)
#7 and 13 samples gives the best scores

for i in range(30, 80, 1):
    monte_carlo_scores(7, (i /100))


    

'''
for 7 samples and 0.7 threshold we have: precision0.6538461538461539,recall 0.17480719794344474,f1 0.27586206896551724
for 7 samples and 0.71 threshold we have: precision0.6744186046511628,recall 0.14910025706940874,f1 0.24421052631578946
for 7 samples and 0.72 threshold we have: precision0.6835443037974683,recall 0.13881748071979436,f1 0.23076923076923078
for 7 samples and 0.73 threshold we have: precision0.746031746031746,recall 0.12082262210796915,f1 0.20796460176991152
for 7 samples and 0.74 threshold we have: precision0.6610169491525424,recall 0.10025706940874037,f1 0.17410714285714288
for 7 samples and 0.75 threshold we have: precision0.7755102040816326,recall 0.09768637532133675,f1 0.1735159817351598
'''
for i in range(30, 80, 1):
    monte_carlo_scores(13, (i /100))

#worse than 7 samples
#so for now we keep 7 samples and 0.73 threshold
NN_preds = NN_monte_carlo_pred(7, 0.73)

        



# we make one final predictor.
#This will hard vote the predictions of our previous voting predictor, the monte carlo NN and the RFC , with the thresholds that we found optimal
final_predictions = []
NN_preds
for i in range(len(predict_thres_RFC(0.55))):
    j = NN_preds[i] + voting_clf_preds(0.51)[i] + predict_thres_RFC(0.55)[i]
    if j > 1.5:
        final_predictions.append(1)
    if j < 1.5:
        final_predictions.append(0)
            
len(final_predictions)

print(f"we have: precision{precision_score (y_test_final, final_predictions)},recall {recall_score(y_test_final, final_predictions)},f1 {f1_score(y_test_final, final_predictions)}")
#we have: precision0.6153846153846154,recall 0.14395886889460155,f1 0.23333333333333336

#lets make a prediction function

def NN_monte_carlo_pred_general(test_set):
    y_probas = np.stack([top_mod_NN(test_set, training=True) for sample in range(7)])
    y_proba = y_probas.mean(axis=0)
    monte_carlo_pred = []
    for i in y_proba:
        if i > (0.73):
            monte_carlo_pred.append(1)
        if i < (0.73):
            monte_carlo_pred.append(0)
    return monte_carlo_pred



def voting_clf_gen(test_set):
    probas = voting_clf.predict_proba(test_set)
    win_one = probas[:, 1]
    predictions = []
    for i in range(len(test_set)):
        if win_one[i] > (0.51):
            predictions.append(1)
        if win_one[i] < (0.51):
            predictions.append(0)
    return predictions



def RFC_gen(test_set):
    y_probas = top_mod_RFC.predict_proba(test_set)
    win_one = y_probas[:, 1]
    thres_preds = []
    for i in range(len(test_set)):
        if win_one[i] > 0.55:
            thres_preds.append(1)
        if win_one[i] < 0.55:
            thres_preds.append(0)
    return(thres_preds)


def final_predictor(test_set):
    final_predictions = []
    NN_preds_gen = NN_monte_carlo_pred_general(test_set=test_set)
    voting_predictor = voting_clf_gen(test_set=test_set)
    RFC_predictor = RFC_gen(test_set=test_set)
    for i in range(len(test_set)):
        j = NN_preds_gen[i] + voting_predictor[i] + RFC_predictor[i]
        if j > 1.5:
            final_predictions.append(1)
        if j < 1.5:
            final_predictions.append(0)
    return final_predictions


print(f"we have: precision{precision_score (y_test_final, final_predictor(X_test_pca))},recall {recall_score(y_test_final, final_predictor(X_test_pca))},f1 {f1_score(y_test_final, final_predictor(X_test_pca))}")
#Winner 1 means that the blue corner will win and 0 means the red
#Our classifier has this precision and recall only for predicting the blue corner winner outcome and not the red one

winner_red_corner = (y_test_final == 0)

j = 0 
k = 0
for i in range(len(y_test_final)):
    if (final_predictions[i] == 0):
        k = k + 1
        if winner_red_corner[i]:
            j = j + 1

print(j / k)
#precision for red corner win (0.63%)

print(j / winner_red_corner.sum())
#recall for red corner win (0.94%)
#that is can not be valid.It is happening because there is an unbalance between the number of instances on the 2 classes on the test test.Lets make a fair test set


(y_test_final == 0).sum() / len(y_test_final)
(y_test_final == 1).sum()
y_test_fair = y_test_final.copy()
X_test_fair_pca = X_test_pca.copy()
X_test_fair_pca_cat = X_test_pca_cat.copy()
i = 0
         
for index in y_test_fair.index:
    if (y_test_fair[index] == 0.0):
        i = i + 1
        if (i < 202):
            y_test_fair.drop(index, inplace=True)
            X_test_fair_pca.drop(index, inplace=True)
            X_test_fair_pca_cat.drop(index, inplace=True)

(y_test_fair == 0).sum() / (y_test_fair == 1).sum()
#the _fair X and y test set have the same number of instances classified as 0 and 1

pred_for_fair_test_set = final_predictor(X_test_fair_pca)
print(f"we have: precision{precision_score (y_test_fair, final_predictor(X_test_fair_pca))},recall {recall_score(y_test_fair, final_predictor(X_test_fair_pca))},f1 {f1_score(y_test_fair, final_predictor(X_test_fair_pca))}")

#we have: precision0.725,recall 0.15167095115681234,f1 0.24733475479744138


winner_red_corner_fair = (y_test_fair == 0)

j = 0 
k = 0
for i in range(len(y_test_fair.reset_index())):
    if (pred_for_fair_test_set[i] == 0):
        k = k + 1
        if (winner_red_corner_fair.reset_index()['Winner'][i]):
            j = j + 1

print(j / k)
#precision for red corner win (0.52%)

print(j / winner_red_corner_fair.sum())
#recall for red corner win (0.95%)

#So we clearly see now that the model is not fit to accurately predict the red corner winner in this test set
