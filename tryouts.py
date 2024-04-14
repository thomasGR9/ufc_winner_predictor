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

X_train_full_1 = pd.read_csv('./X_train_concat_branch.csv', dtype=np.float64)
X_test_full_1 = pd.read_csv('./X_test_concat_branch.csv', dtype=np.float64)
y_train_1 = pd.read_csv('./y_train_branch.csv', dtype=np.float64)
y_test_1 = pd.read_csv('./y_test_branch.csv', dtype=np.float64)





X_train_without_pca_full = pd.read_csv('./X_train_num_scaled_branch.csv', dtype=np.float64)
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


(y_train_1 == 1).sum() / (y_train_1 == 0).sum()
(y_test_1 == 1).sum() / (y_test_1 == 0).sum()

#we will make it 1

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

y_probas = cross_val_predict(top_mod_RFC, X_train_pca, y_train_final, cv=3, method="predict_proba")

len(y_probas[:, 1])

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
    return print(f"for{threshold} threshold we have: precision{precision_score (y_train_final, preds)},recall {recall_score(y_train_final, preds)},f1 {f1_score(y_train_final, preds)}")

for i in range(30, 80, 1):
    RFC_test_scores(i / 100)

'''
for0.54 threshold we have: precision0.5823353293413174,recall 0.26516700749829586,f1 0.36440281030444965
for0.55 threshold we have: precision0.5914221218961625,recall 0.17859577368779822,f1 0.27434554973821984
for0.56 threshold we have: precision0.6037037037037037,recall 0.1111111111111111,f1 0.1876799078871618

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

'''voting_clf.fit(X_train_pca, y_train_final)
voting_clf.estimators_
predictions_vot_clf = voting_clf.predict(X_test_pca)
for name, clf in voting_clf.named_estimators_.items():
    print(name, "= prec", precision_score(y_test_final, clf.predict(X_test_pca)), ", recall", recall_score(y_test_final, clf.predict(X_test_pca)), ", f1_score", f1_score(y_test_final, clf.predict(X_test_pca)))
#we see that the SVC class is a very weak learner and we should replace it with something better

    
(predictions_vot_clf == y_test_final).sum() / len(y_test_final)
pres_test = precision_score(y_test_final, predictions_vot_clf)
recall_test = recall_score(y_test_final, predictions_vot_clf)
f1_test = f1_score(y_test_final, predictions_vot_clf)
print(f"The voting cglf has default precision score of {pres_test} recall score of{recall_test} and f1 score of {f1_test}")
'''
#this function will print the average classification scores of a model on 3 different test sets (who have equal number of negative and positive classes).The model is fitted on a different training set (would be x - test set) that 3 times 
def probas_stratify_kfolds(model, x, y, threshold):
    model = model
    kf = StratifiedKFold(n_splits=3)
    probas = []
    mean_probas = []
    precisions = []
    recalls = []
    f1s = []
    for idx in kf.split(X=x, y=y):
        predictions = []
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict_proba(xtest)
        win_one_proba = preds[:, 1]
        for i in range(len(xtest)):
            if win_one_proba[i] > (threshold):
                predictions.append(1)
            if win_one_proba[i] < (threshold):
                predictions.append(0) 
        precision = precision_score(ytest, predictions)
        recall = recall_score(ytest, predictions)
        f1 = f1_score(ytest, predictions)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    avg_prec = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    return print(f"for{threshold} threshold we have: precision{avg_prec},recall {avg_recall},f1 {avg_f1}")

for i in range(40, 55, 1):
    probas_stratify_kfolds(voting_clf, x=X_train_pca, y=y_train_final, threshold=i / 100)
'''
for0.5 threshold we have: precision0.5769526287754135,recall 0.17998421211488008,f1 0.27341285833782764
for0.51 threshold we have: precision0.5746671876600384,recall 0.13025933923057276,f1 0.2098909368582988
'''

for i in range(50, 60, 1):
    probas_stratify_kfolds(top_mod_RFC, x=X_train_pca, y=y_train_final, threshold=i / 100)

'''
for0.54 threshold we have: precision0.5732706619663142,recall 0.2666039794757493,f1 0.36167421899877217
for0.55 threshold we have: precision0.5991514621299199,recall 0.17863295488844003,f1 0.273539911778129
for0.56 threshold we have: precision0.6175968595323433,recall 0.11182698335306818,f1 0.188151075833639
'''
#we will make a modified version of the above function for the NN since we cant train it (fit it)
def probas_stratify_kfolds_NN(model, x, y, threshold, samples):
    model = model
    kf = StratifiedKFold(n_splits=3)
    probas = []
    mean_probas = []
    precisions = []
    recalls = []
    f1s = []
    for idx in kf.split(X=x, y=y):
        predictions = []
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x.loc[train_idx]
        ytrain = y.loc[train_idx]
        
        xtest = x.loc[test_idx]
        ytest = y.loc[test_idx]
        
        
        preds = np.stack([model(xtest, training=True) for sample in range(samples)])
        win_one_proba = preds.mean(axis=0)
        for i in range(len(xtest)):
            if win_one_proba[i] > (threshold):
                predictions.append(1)
            if win_one_proba[i] < (threshold):
                predictions.append(0) 
        precision = precision_score(ytest, predictions)
        recall = recall_score(ytest, predictions)
        f1 = f1_score(ytest, predictions)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    avg_prec = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    return print(f"for{threshold} threshold and samples {samples} we have: precision{avg_prec},recall {avg_recall},f1 {avg_f1}")


for i in range(65, 80, 1):
    for j in range(2, 15, 1):
        probas_stratify_kfolds_NN(top_mod_NN, x=X_train_pca, y=y_train_final, threshold=i / 100, samples=j)

'''
for0.74 threshold and samples 3 we have: precision0.809598471149104,recall 0.1301976823449216,f1 0.22431388245012188
'''



'''
for0.7 threshold we have: precision0.7661371563333953,recall 0.19768234492160874,f1 0.31423857273636924
for0.71 threshold we have: precision0.7589078037932596,recall 0.17450579413769596,f1 0.28359892451964064
for0.72 threshold we have: precision0.7688079522237938,recall 0.158145875937287,f1 0.2622797583249561
for0.73 threshold we have: precision0.7803480942938973,recall 0.13769597818677573,f1 0.2340723950986420
'''

for i in range(2, 15, 1):
    probas_stratify_kfolds_NN(top_mod_NN, x=X_train_pca, y=y_train_final, threshold=0.71, samples=i)
'''
probas_voting_clf = cross_val_predict(voting_clf, X_train_pca, y_train_final, cv=3, method="predict_proba")

win_one_voting = probas_voting_clf[:, 1]

def voting_clf_preds(threshold):
    predictions = []
    for i in range(len(X_train_pca)):
        if win_one_voting[i] > (threshold):
            predictions.append(1)
        if win_one_voting[i] < (threshold):
            predictions.append(0)
    return predictions

len(voting_clf_preds(0.5))

def voting_scores(threshold):
    preds = voting_clf_preds(threshold=threshold)
    return print(f"for{threshold} threshold we have: precision{precision_score (y_train_final, preds)},recall {recall_score(y_train_final, preds)},f1 {f1_score(y_train_final, preds)}")
    
for i in range(30, 80, 1):
    voting_scores(i / 100)

#for0.51 threshold we have: precision0.5984555984555985,recall 0.1056578050443081,f1 0.17960602549246812
'''
voting_scores(0.50)
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

NN_monte_carlo_pred(7, 0.7)

def monte_carlo_scores(samples, threshold):
    preds = NN_monte_carlo_pred(samples, threshold)
    return print(f"for {samples} samples and {threshold} threshold we have: precision{precision_score (y_test_final, preds)},recall {recall_score(y_test_final, preds)},f1 {f1_score(y_test_final, preds)}")

monte_carlo_scores(7, 0.7)

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
    y_probas = np.stack([top_mod_NN(test_set, training=True) for sample in range(3)])
    y_proba = y_probas.mean(axis=0)
    monte_carlo_pred = []
    for i in y_proba:
        if i > (0.73):
            monte_carlo_pred.append(1)
        if i < (0.73):
            monte_carlo_pred.append(0)
    return monte_carlo_pred

len(NN_monte_carlo_pred_general(X_test_pca))

def voting_clf_gen(test_set):
    probas = voting_clf.predict_proba(test_set)
    win_one = probas[:, 1]
    predictions = []
    for i in range(len(test_set)):
        if win_one[i] > (0.50):
            predictions.append(1)
        if win_one[i] < (0.50):
            predictions.append(0)
    return predictions

(RFC_gen(X_test_pca))

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

final_predictor(X_test_fair_pca)


print(f"we have: precision{precision_score (y_test_fair, final_predictor(X_test_fair_pca))},recall {recall_score(y_test_fair, final_predictor(X_test_fair_pca))},f1 {f1_score(y_test_fair, final_predictor(X_test_fair_pca))}")
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
def scores_for_neg_class(X_test, y_test):
    pred_for_test_set = final_predictor(X_test)
    winner_red_corner = (y_test == 0)
    j = 0 
    k = 0
    for i in range(len(y_test.reset_index())):
        if (pred_for_test_set[i] == 0):
            k = k + 1
            if (winner_red_corner.reset_index()['Winner'][i]):
                j = j + 1
    precision = j / k
    recall = j / winner_red_corner.sum()
    f1 = 2 * ((precision * recall) / (precision + recall))
    return print(f"we have: precision {precision},recall {recall},f1 {f1}")

scores_for_neg_class(X_test_fair_pca, y_test_fair)
#we have: precision 0.5401662049861495,recall 1.0,f1 0.7014388489208633

#So we clearly see now that the model is not fit to accurately predict the red corner winner (with our required precision) in this test set
#This is happening because we append 1 only to the very confident predictions about this class but we label 0 all the other without concern about the confidence
#Considering that in reality the outcome of an ufc fight should not rely on witch corner the fighter is on (red or blue) it must be valid to use the same confidence thresholds for the 2 labels of the predict proba methods
#So now we will configure the functions we defined above (for the final predictor) such as we have the same precision scores for the two predicted labels 
#We will do this by appending 1 with the predict proba threshold we currently have, but will append 0 if the predict proba for this class is lower than (1 - threshold_for_one)
#All the rest predictions that their predict proba is low for the precision we aim will be imputed by some text like (not sure enough)
#That approach will make the predictor output this text most of the times, but when it gives a class prediction we can expect that it is very confident about it


def NN_monte_carlo_pred_general_negative(test_set):
    y_probas = np.stack([top_mod_NN(test_set, training=True) for sample in range(7)])
    y_proba = y_probas.mean(axis=0)
    monte_carlo_pred = []
    for i in y_proba:
        if i > (0.73):
            monte_carlo_pred.append(1)
        if i < (0.73):
            monte_carlo_pred.append(0)
    return monte_carlo_pred







def model_gen_negative(model, test_set, threshold):
    y_probas = model.predict_proba(test_set)
    win_zero = y_probas[:, 0]
    thres_preds = []
    for i in range(len(test_set)):
        if win_zero[i] > (threshold):
            thres_preds.append(0)
        if win_zero[i] < (threshold):
            thres_preds.append(1)
    return(thres_preds)
for i in range(30, 50, 1):
    print(f"for threshold {i / 100} we have: precision{precision_score(y_test_fair, model_gen_negative(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=i / 100))},recall {recall_score(y_test_fair, model_gen_negative(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=i / 100))},f1 {f1_score(y_test_fair, model_gen_negative(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=i / 100))}")    


def model_gen_test(model, test_set, threshold):
    y_probas = model.predict_proba(test_set)
    win_zero = y_probas[:, 1]
    thres_preds = []
    for i in range(len(test_set)):
        if win_zero[i] >= (threshold):
            thres_preds.append(1)
        if win_zero[i] <= (1 - threshold):
            thres_preds.append(0)
        elif ((win_zero[i] < (threshold)) and (win_zero[i] > (1 - threshold))):
            thres_preds.append(-1)
    return(thres_preds)

        


def metrics_for_both_classes(model, test_set_x, test_set_y, threshold):
    preds_test = model_gen_test(model = model, test_set = test_set_x, threshold = threshold)
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
    precision_zero = correct_zero / predicted_zero
    recall_zero = correct_zero / winner_zero.sum()
    precision_one = correct_one / predicted_one
    recall_one = correct_one / winner_one.sum()
    percentage_of_none = predicted_none / len(test_set_y)
    return print(f"for threshold {threshold}\nprecision for zero is {round(precision_zero, 3)} and recall for zero is {round(recall_zero, 3)} \nprecision for one is {round(precision_one, 3)} and recall for one is {round(recall_one, 3)}\npercentage of none is {round(percentage_of_none, 3)}\nnumber of zero predictions{predicted_zero}\nnumber of predicted one {predicted_one}\n\n")


for i in range(50, 60, 1):
    metrics_for_both_classes(voting_clf, test_set_x=X_test_fair_pca, test_set_y=y_test_fair, threshold= i / 100)

#we see that the voting_clf model is vary bias towards predicting the zero class (thats because of the big inbalance in the training set)
 
        


print(f"we have: precision{precision_score(y_test_fair, model_gen_test(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=0.55))},recall {recall_score(y_test_fair, model_gen_test(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=0.55))},f1 {f1_score(y_test_fair, model_gen_test(model = top_mod_RFC , test_set=X_test_fair_pca, threshold=0.55))}")

def final_predictor_negative(test_set):
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


y_probas = np.stack([top_mod_NN(X_test_pca, training=True) for sample in range(7)])
y_proba = y_probas.mean(axis=0)

voting_clf_gen_negative(X_test_pca)
RFC_gen_negative(X_test_pca)