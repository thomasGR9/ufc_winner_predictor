import pandas as pd
import numpy as np

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

import tensorflow as tf

n_units = 300
activation = tf.keras.activations.gelu
initializer = tf.keras.initializers.he_normal()


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train_pca.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_units, activation= activation, kernel_initializer=initializer),
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
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])




lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1 , decay_steps=10000, decay_rate=0.96 )
optimizer = tf.keras.optimizers.Adam(clipnorm=1, learning_rate=lr_schedule)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy' ,patience=20, restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#we should put a higher weight in the underpressented class (1 in this case)
weight_minoniry_class = (y_train_final == 0).sum() / (y_train_final == 1).sum()

history = model.fit(X_train_pca, y_train_final, epochs=100, validation_data=(X_valid_pca, y_valid_final), batch_size=32, callbacks=[callback], class_weight= {0:1, 1:weight_minoniry_class})

model.evaluate(x=X_test_pca, y=y_test_final)

pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot(figsize=(8, 5))

#monte carlo
y_probas = np.stack([model(X_test_final, training=True) for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_proba
monte_carlo_pred = []
for i in y_proba:
    if i > 0.5:
        monte_carlo_pred.append(1)
    if i < 0.5:
        monte_carlo_pred.append(0)
    
(monte_carlo_pred == y_test_final).sum() / len(y_test_final)

from keras.models import load_model
'''
model.save("62acc.h5") 62%acc, X with pca and cat
'''

top_mod_NN = tf.keras.models.load_model('./62acc.h5')
top_mod_NN.summary()
top_mod_NN.evaluate(x=X_test_pca_cat, y=y_test_final)

from sklearn.ensemble import RandomForestClassifier
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
#the best test accuracy is  59.59% for n_est:400 and max_depth:6
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import precision_score


def optimize(params, param_names, x, y):
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

from skopt import space

param_space = [
    space.Integer(3, 15, name="max_depth"),
    space.Integer(200, 700, name="n_estimators"),
    space.Categorical(["gini", "entropy"], name="criterion"),
    space.Real(0.01, 1, prior="uniform" ,name="max_features")
]
param_names = [
    "max_depth",
    "n_estimators",
    "criterion",
    "max_features"
]

from functools import partial
optimization_function = partial(
    optimize,
    param_names = param_names,
    x = X_train_pca_full,
    y = y_train_full
)
from skopt import gp_minimize
result =  gp_minimize(
    optimization_function,
    dimensions = param_space,
    verbose=10 
)

print(
    dict(zip(param_names, result.x))
)


top_mod_RFC = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, class_weight={0:1 , 1:class_weight})
top_mod_RFC.fit(X_train_pca_full, y_train_full)
(top_mod_RFC.predict(X_test_pca) == y_test_final).sum() / y_test_final.shape[0]




from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(max_depth=6, learning_rate=0.05, n_estimators=500, n_iter_no_change=10,random_state=42)
gbcl.fit(X_train_pca_full, y_train_full)
gbcl.n_estimators_
(gbcl.predict(X_train_pca) == y_train_final).sum() / X_train_pca.shape[0]
(gbcl.predict(X_test_pca) == y_test_final).sum() / y_test_final.shape[0]
#61

#we should consider that is possible that the models performance is overestimated because of the inbalance of the two classes populations  
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
(gbcl.predict(X_test_fair_pca) == y_test_fair).sum() / y_test_fair.shape[0]
#55%
(top_mod_RFC.predict(X_test_fair_pca) == y_test_fair).sum() / y_test_fair.shape[0]
#59%
top_mod_NN.evaluate(x=X_test_fair_pca_cat, y=y_test_fair)
#58%

#We see that the performance of the gbcl predictor (that doesnt have a class_weight parameter) drop significantly
'''
#Its time to choose the right balance between precision and recall.I believe that the metric that i should give the most of the weight is precision.
#I look at it as a gambler's view.Its way more important that you are more certain about the correct winner than finding the most correct winners, if you are gonna bet for the winner.
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

(top_mod_RFC.predict(X_test_fair_pca) == 1).sum() 
#manually calculating precision score to be sure
top_mod_RFC.predict(X_test_fair_pca) == 1
j = 0
for i in range(X_test_fair_pca.shape[0]):
    if (top_mod_RFC.predict(X_test_fair_pca) == 1)[i]:
        if (y_test_fair == 1).tolist()[i]:
            j = j + 1

print(j / (top_mod_RFC.predict(X_test_fair_pca) == 1).sum())


RFC_winner_one_pred = (top_mod_RFC.predict(X_test_fair_pca) == 1)
y_test_winner_one = (y_test_fair == 1)

confusion_matrix(y_test_winner_one, RFC_winner_one_pred)
precision_score(y_test_winner_one, RFC_winner_one_pred)
recall_score(y_test_winner_one, RFC_winner_one_pred)
from sklearn.model_selection import cross_val_predict
#Winner 0 means red corner winner 1 means blue corner winner

y_probas = cross_val_predict(top_mod_RFC, X_test_fair_pca, y_test_fair, cv=3, method="predict_proba")
y_scores = y_probas[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test_fair, y_scores)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.show()

precisions[:-1].max()
idx_for_80_prec = (precisions > 0.80).argmax()
est_prob_thresh_win_one = thresholds[idx_for_80_prec]
y_pred_80_prec = (y_scores >= est_prob_thresh_win_one)
precision_score(y_test_fair, y_pred_80_prec)
#81%
recall_score(y_test_fair, y_pred_80_prec)
#7%

#We see that if we trust this classifier where he outputs an estimated probability over 67% for the blue to win, we will have about 80% precision 
#So for the test set if this classifier predicts that red is gonna win,there is about 80%chance it is correct but it will be able to predict about the 7% of all the winners
#But because a very low estimated probability means a high precision that red is gonna win,where we can also bet on.

precisions_2, recalls_2, thresholds_2 = precision_recall_curve(y_test_fair, y_probas[:, 0])
idx_for_80_prec_2 = (precisions_2 > 0.70).argmax()
est_prob_thresh_loss_one = thresholds_2[idx_for_80_prec_2]
y_pred_80_prec_2 = (y_probas[:, 0] >= est_prob_thresh_loss_one)
precision_score(y_test_fair, y_pred_80_prec_2)
#80%
recall_score(y_test_fair, y_pred_80_prec_2)
#1%

#we see that in order to have over 80% precision on the classifier for the blue to lose we need below 1-est_prob_thresh_loss_one = 0.36 on our y_scores list (the classifier estimated probabilities)

#all in all if  y_score>0.67 or y_score<0.36 we can trust it with about 80% precision for the blue corner to win or lose 
'''