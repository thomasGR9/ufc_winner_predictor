import pandas as pd 
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import re

df = pd.read_csv('../../datasets/ufc-master.csv')
for column in df.columns:
    print(column)



#dropping a bunch of columns that should not matter much
df1 = df[df.columns[9:]].drop(['title_bout', 'empty_arena', 'constant_1', 'B_match_weightclass_rank', 'R_match_weightclass_rank', 'R_Pound-for-Pound_rank', 'B_Pound-for-Pound_rank', 'better_rank'], axis=1)

for column in df1.columns:
    print(column)

df2 = df1[df1.columns[:91]]

for column in df1.columns:
    print(column)

i = 0
for column in df2.columns:
    i = i + 1
    if column == 'R_Women\'s Flyweight_rank':
        print(i)
 
l = 0    
for j in range(df2.shape[0]):
    f = 0
    for column in df2.iloc[:, 68:].columns:
        if df2.iloc[:, 68:].isna().iloc[j][column]:
            f = f + 1
    if f < 23:
        l = l + 1

print(f'percentage of the dataset that fights have ranked fighters is: {(l/df2.shape[0])*100}%')



[col for col in df2.iloc[:, 68:79].columns]
~df2.iloc[:, 68:79].isna().iloc[0]
~df2.isna().iloc[0]['B_Flyweight_rank']
#We have a column for every weight class and True-False statements on where the fighters are
R_Series = []
for j in range(df2.shape[0]):
    R_Series.append(16)

R_Series

for j in range(df2.shape[0]):
    for column in df2.iloc[:, 68:79].columns:
        if ~df2.iloc[:, 68:79].isna().iloc[j][column]:
            R_Series[j] = df2.iloc[:, 68:79].iloc[j][column]
   
R_Series        
#The R_Series list will have the rank of the fighter of the division he is ,if he is unranked it will append 16 (ufc ranking rank as high as 15)    
~df2.iloc[:, 79:].isna().iloc[0]

B_Series = []
for j in range(df2.shape[0]):
    B_Series.append(16)

B_Series

for j in range(df2.shape[0]):
    for column in df2.iloc[:, 79:].columns:
        if ~df2.iloc[:, 79:].isna().iloc[j][column]:
            B_Series[j] = df2.iloc[:, 79:].iloc[j][column]
        
   
B_Series
#same as R but for fighters on the B (blue) corner

df2['R_rank'] = R_Series
df2['R_rank']
df2['B_rank'] = B_Series
df2['B_rank']
df2['B_rank'].replace(0, 16, inplace=True)
df3 = df2.copy()
df3.columns

i = 0
for column in df3.columns:
    i = i + 1
    if column == 'lose_streak_dif':
        print(i)

df3.iloc[:, 52:67].columns
df3.shape[0]
df3['B_current_lose_streak'].iloc[4891]
df3['R_current_lose_streak'].iloc[4891]
df3['lose_streak_dif'].iloc[4891]
#some diff columns are B - R but some others are mixed
(df3['B_current_lose_streak'] - df3['R_current_lose_streak'] == df3['lose_streak_dif']).sum()
#mixed
(df3['B_current_win_streak'] - df3['R_current_win_streak'] == df3['win_streak_dif']).sum()
#B - R
(df3['longest_win_streak_dif'] == df3['B_longest_win_streak'] - df3['R_longest_win_streak']).sum()
#B-R
(df3['win_dif'] == df3['B_wins'] - df3['R_wins']).sum()
#B - R
(df3['loss_dif'] == df3['B_losses'] - df3['R_losses']).sum()
#mixed
(df3['total_round_dif'] == df3['B_total_rounds_fought'] - df3['R_total_rounds_fought']).sum()
#B - R
(df3['total_title_bout_dif'] == df3['B_total_title_bouts'] - df3['R_total_title_bouts']).sum()
#B - R
(df3['ko_dif'] == df3['B_win_by_KO/TKO'] - df3['R_win_by_KO/TKO']).sum()
#mixed
(df3['sub_dif'] == df3['B_win_by_Submission'] - df3['R_win_by_Submission']).sum()
#B-R
(df3['height_dif'] == df3['B_Height_cms'] - df3['R_Height_cms']).sum()
#mixed
(df3['reach_dif'] == df3['B_Reach_cms'] - df3['R_Reach_cms']).sum()
#mixed
(df3['age_dif'] == df3['B_age'] - df3['R_age']).sum()
#mixed
(df3['sig_str_dif'] == df3['B_avg_SIG_STR_landed'] - df3['R_avg_SIG_STR_landed']).sum()
#mixed    
(df3['avg_sub_att_dif'] == df3['B_avg_SUB_ATT'] - df3['R_avg_SUB_ATT']).sum()
#mixed    
(df3['avg_td_dif'] == df3['B_avg_TD_landed'] - df3['R_avg_TD_landed']).sum()
#mixed 
k = 0
j = 0
for i in range(df3.shape[0]):
    if ~(df3['B_current_lose_streak'] - df3['R_current_lose_streak'] == df3['lose_streak_dif'])[i]:
        k = k + 1
        if (df3['R_current_lose_streak'] - df3['B_current_lose_streak'] == df3['lose_streak_dif'])[i]:
            j = j + 1

k / j
#so when the diff is not B - R s R - B and is not a mistake
for i in range(df3.shape[0]):
    if ~(df3['B_current_lose_streak'] - df3['R_current_lose_streak'] == df3['lose_streak_dif'])[i]:
        if (df3['R_current_lose_streak'] - df3['B_current_lose_streak'] == df3['lose_streak_dif'])[i]:
            df3['lose_streak_dif'][i] = - df3['lose_streak_dif'][i]
           

(df3['B_current_lose_streak'] - df3['R_current_lose_streak'] == df3['lose_streak_dif']).sum()

mixed_dif = ['loss_dif', 'ko_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif']
mixed_B = ['B_losses', 'B_win_by_KO/TKO', 'B_Height_cms', 'B_Reach_cms', 'B_age', 'B_avg_SIG_STR_landed', 'B_avg_SUB_ATT', 'B_avg_TD_landed']
mixed_R = ['R_losses', 'R_win_by_KO/TKO', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'R_avg_SIG_STR_landed', 'R_avg_SUB_ATT', 'R_avg_TD_landed']

for j in range(len(mixed_dif)):
    print(j)
    for i in range(df3.shape[0]):
        if ~(df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]])[i]:
            if (df3[mixed_R[j]] - df3[mixed_B[j]] == df3[mixed_dif[j]])[i]:
                df3[mixed_dif[j]][i] = - df3[mixed_dif[j]][i]
    print((df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]]).sum())

#so not only the first and the fifth dif fixed


for j in range(len(mixed_dif)):
    k = 0
    l = 0
    print(j)
    for i in range(df3.shape[0]):
        if ~(df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]])[i]:
            k = k + 1
            if (df3[mixed_R[j]] - df3[mixed_B[j]] == df3[mixed_dif[j]])[i]:
                l = l + 1
    print(f"k is {k}, l is {l}")

#So we verify that the occurances where the diff is not B - R is not R - B either, lets see what it is
k = 0
for i in range(df3.shape[0]):
    if ~(df3['ko_dif'] == df3['B_win_by_KO/TKO'] - df3['R_win_by_KO/TKO'])[i]:
        k = k + 1
        print(k)
        print(f"diff is {df3['ko_dif'][i]}, B is {df3['B_win_by_KO/TKO'][i]}, R is {df3['R_win_by_KO/TKO'][i]}")

(df3['ko_dif'] == df3['B_win_by_KO/TKO'] - df3['R_win_by_KO/TKO']).sum()
~(df3['R_win_by_TKO_Doctor_Stoppage'] == 0).sum() 
~(df3['R_win_by_KO/TKO'] == 0).sum() 
~(df3['ko_dif'] == df3['B_win_by_TKO_Doctor_Stoppage'] - df3['R_win_by_TKO_Doctor_Stoppage']).sum()
#is not just different columns
for j in range(len(mixed_dif)):
    print(j)
    for i in range(df3.shape[0]):
        if ~(df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]])[i]:
            df3[mixed_dif[j]][i] = df3[mixed_B[j]][i] - df3[mixed_R[j]][i] 
    print((df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]]).sum())

for j in range(len(mixed_dif)):
    print((df3[mixed_B[j]] - df3[mixed_R[j]] == df3[mixed_dif[j]]).sum())
    print((df3[mixed_R[j]] - df3[mixed_B[j]] == df3[mixed_dif[j]]).sum())

#the last 3 columns of this list is not working maybe because of nans?
for j in range(5, 8):
    print(f"no of nans in {mixed_B[j]} is {df3[mixed_B[j]].isna().sum()}")
    print(f"no of nans in {mixed_R[j]} is {df3[mixed_R[j]].isna().sum()}")  

#yes they have nans, we will impute the B and R columns with the mean, and the diff column with 0

df5 = df3.copy()



values = {"sig_str_dif": 0,"avg_sub_att_dif": 0,"avg_td_dif": 0}
df5.fillna(value=values, inplace=True)
mean_list = ['B_avg_SIG_STR_landed', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'R_avg_SIG_STR_landed', 'R_avg_SUB_ATT', 'R_avg_TD_landed']
for j in range(len(mean_list)):
    df5[mean_list[j]].fillna(int(df5[mean_list[j]].mean()), inplace=True)
    
df5[mean_list[0]].isna().sum()
df5['sig_str_dif']

df5[mean_list].mean()


df5.columns
df5['Rank_dif'] = df5['B_rank'] - df5['R_rank']
(df5['Rank_dif'] < 0).sum()
#we will then drop the B and R rank because we filled the nan values with 16 witch is not true and just served for the dif

df5.drop(["R_rank" ,"B_rank"], axis=1, inplace=True)

len(df5.columns)
for i in range(len(df5.columns)):
    if df5.columns[i] == 'R_Women\'s Flyweight_rank':
        print(i)

rank_columns = []
for i in range(67, 91):
    rank_columns.append(df5.columns[i])

rank_columns

df5.drop(rank_columns, axis=1, inplace=True)

df5.columns

#we will see the remaining columns with Nans

for column in df5.columns:
    if (~df5[column].isna()).sum() < 4896:
        print(column)

df5['B_Stance'].value_counts()

(df5['B_Stance'] == 'Switch').sum()
(df5['B_Stance'] == 'Switch ').sum()

df5.replace({'B_Stance': 'Switch '}, 'Switch', inplace=True)
df5['B_Stance'].value_counts()
df5.replace({'B_Stance': 'Open Stance'}, 'Orthodox', inplace=True)
df5['B_Stance'].value_counts()
df5['B_Stance'].fillna('Orthodox', inplace=True)
df5['B_Stance'].isna().sum()

df5['R_Stance'].value_counts()
df5.replace({'R_Stance': 'Open Stance'}, 'Orthodox', inplace=True)
df5['R_Stance'].value_counts()
df5['R_Stance'].isna().sum()
#for the missing values of SIG_STR_pct we will first collect the values of SIG_STR_landed with the same indexes witch are available

indexes_of_nans = []
for i in range(df5.shape[0]):
    if df5['B_avg_SIG_STR_pct'].isna()[i]:
        indexes_of_nans.append(i)

len(indexes_of_nans)
sig_str_landed_of_nan_pct = []

for index in indexes_of_nans:
    sig_str_landed_of_nan_pct.append(df5['B_avg_SIG_STR_landed'][index])

len(sig_str_landed_of_nan_pct)
j=0
for i in range(len(sig_str_landed_of_nan_pct)):
    if sig_str_landed_of_nan_pct[i] == 26:
        j = j + 1
        
print(j)
#all values are 26
(df5['B_avg_SIG_STR_landed'] == 26).sum()
#we will impute the nans of the pct column with the mean of this column for the fighters that have avg_sig_str_landed = 26 and not nan pct.We feel that is the most representative value
pct_mean_impute = df5[df5['B_avg_SIG_STR_landed'] == 26]['B_avg_SIG_STR_pct'].mean(skipna=True)
df5['B_avg_SIG_STR_pct'].fillna(pct_mean_impute, inplace=True)
df5['B_avg_SIG_STR_pct'].isna().sum()

df5['R_avg_SIG_STR_pct'].isna().sum()
indexes_of_nans_R = []
for i in range(df5.shape[0]):
    if df5['R_avg_SIG_STR_pct'].isna()[i]:
        indexes_of_nans_R.append(i)

indexes_of_nans_R
sig_str_landed_of_nan_pct_R = []
for index in indexes_of_nans_R:
    sig_str_landed_of_nan_pct_R.append(df5['R_avg_SIG_STR_landed'][index])

sig_str_landed_of_nan_pct_R
len(sig_str_landed_of_nan_pct_R)
j=0
for i in range(len(sig_str_landed_of_nan_pct_R)):
    if sig_str_landed_of_nan_pct_R[i] == 27:
        j = j + 1

print(j)
#so all the values are nans,so as before we will impute all the nans of the pct column with the mean of the remaining pct of all the figters with 27 sig_str_landed
(df5['R_avg_SIG_STR_landed'] == 27).sum()
pct_mean_impute_R = df5[df5['R_avg_SIG_STR_landed'] == 27]['R_avg_SIG_STR_pct'].mean(skipna=True)
df5['R_avg_SIG_STR_pct'].fillna(pct_mean_impute_R, inplace=True)
df5['R_avg_SIG_STR_pct'].isna().sum()

df5['B_avg_TD_pct'].isna().sum()

nan_td_pct_indexes_B = []
for i in range(df5.shape[0]):
    if df5['B_avg_TD_pct'].isna()[i]:
        nan_td_pct_indexes_B.append(i)
nan_td_pct_indexes_B
len(nan_td_pct_indexes_B)
td_landed_of_nanpct_B = []
for index in nan_td_pct_indexes_B:
    td_landed_of_nanpct_B.append(df5['B_avg_TD_landed'][index])
    
td_landed_of_nanpct_B
l = 0
for value in td_landed_of_nanpct_B:
    l = l + value

print(f"mean value of list is {l / len(td_landed_of_nanpct_B)}")
#is basically 1 so we will impute like the sig_str columns
impute_TD_pct_B = df5[df5['B_avg_TD_landed'] == 1]['B_avg_TD_pct'].mean(skipna=True)
df5['B_avg_TD_pct'].fillna(impute_TD_pct_B, inplace=True)
df5['B_avg_TD_pct'].isna().sum()

df5['R_avg_TD_pct'].isna().sum()

nan_td_pct_indexes_R = []
for i in range(df5.shape[0]):
    if df5['R_avg_TD_pct'].isna()[i]:
        nan_td_pct_indexes_R.append(i)
nan_td_pct_indexes_R
len(nan_td_pct_indexes_R)
td_landed_of_nanpct_R = []
for index in nan_td_pct_indexes_R:
    td_landed_of_nanpct_R.append(df5['R_avg_TD_landed'][index])
    
td_landed_of_nanpct_R
l = 0
for value in td_landed_of_nanpct_R:
    l = l + value

print(f"mean value of list is {l / len(td_landed_of_nanpct_R)}")
#so 1 in this case too,we will impute with the same logic again
impute_TD_pct_R = df5[df5['R_avg_TD_landed'] == 1]['R_avg_TD_pct'].mean(skipna=True)
df5['R_avg_TD_pct'].fillna(impute_TD_pct_R, inplace=True)
df5['B_avg_TD_pct'].isna().sum()

#so we are done with the nans in the dataset


len(df5.columns)

df5.columns
#the columns we will use to predict the winner have relative importance to the gender and weight of the fighters.To avoid biases we will stratify according to weight_class witch combines the two factors by specifying womans and mans weight class
df5['weight_class'].value_counts() 
#so we can drop the gender and weight columns because their information is in the weight class column
df5.drop(['gender', 'B_Weight_lbs', 'R_Weight_lbs'], axis=1, inplace=True)
y = df5['Winner']
x = df5.drop(['Winner'], axis=1)
(y == "Red").sum() / (y == "Blue").sum()
#we came back here to save test and training sets based on stratifying the winner values too

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=x['weight_class'], test_size=0.2, shuffle=True)
(y_train == "Red").sum() / (y_train == "Blue").sum()
(y_test == "Red").sum() / (y_test == "Blue").sum()

df5['weight_class'].value_counts() / df5.shape[0]
X_train['weight_class'].value_counts() / X_train.shape[0]
X_test['weight_class'].value_counts() / X_test.shape[0]
#about the same.Now we can drop the weight class column from test and train dataset since we only needed it for the stratification
X_train.drop(['weight_class'], axis=1, inplace=True)
X_test.drop(['weight_class'], axis=1, inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
len(X_test.columns)
X_test_cat = X_test.select_dtypes(exclude=np.number)
X_train_cat = X_train.select_dtypes(exclude=np.number)
X_test_num = X_test.select_dtypes(include=np.number)
X_train_num = X_train.select_dtypes(include=np.number)
X_train_cat['R_Stance'].value_counts()


cat_encoder = OneHotEncoder()
train_cat_en = cat_encoder.fit_transform(X_train_cat)
X_train_cat_en = pd.DataFrame(train_cat_en.toarray(), columns=cat_encoder.get_feature_names_out(), index=X_train_cat.index )
X_train_cat_en
cat_encoder.transform(X_test_cat).toarray()
X_test_cat_en = pd.DataFrame(cat_encoder.transform(X_test_cat).toarray(), columns=cat_encoder.get_feature_names_out(), index=X_test_cat.index )
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
#We will replace the red with 0 and blue with 1
y_train.replace(['Red', 'Blue'], [0, 1], inplace=True)
y_test.replace(['Red', 'Blue'], [0, 1], inplace=True)


X_train_num.columns
#lets handle the numerical columns

for column in X_train_num.columns:
    X_train_num[column].hist(legend=True)
    plt.show()

X_train_num['B_win_by_Decision_Majority'].value_counts(normalize=True)
X_train_num['R_win_by_Decision_Majority'].value_counts(normalize=True)
j = 0
for i in range(len(X_train_num['B_win_by_Decision_Majority'])):
    if X_train_num['B_win_by_Decision_Majority'][i] == X_train_num['R_win_by_Decision_Majority'][i]:
        j = j + 1
print(f"percentage same wins by majority decision {j / len(X_train['B_win_by_Decision_Majority'])*100}%") 
#so we have this columns with roughly the same values and nearly all of the equal to 0.Its best to drop them
X_test_num.drop(['B_win_by_Decision_Majority', 'R_win_by_Decision_Majority'], axis=1, inplace=True)
X_train_num.drop(['B_win_by_Decision_Majority', 'R_win_by_Decision_Majority'], axis=1, inplace=True)

X_train_num['B_draw'].value_counts(normalize=True)
X_train_num['R_draw'].value_counts(normalize=True)
#the same with the draws
X_test_num.drop(['B_draw', 'R_draw'], axis=1, inplace=True)
X_train_num.drop(['B_draw', 'R_draw'], axis=1, inplace=True)

X_train_num['B_total_title_bouts'].value_counts(normalize=True)
X_train_num['R_total_title_bouts'].value_counts(normalize=True)
X_train_num['B_total_title_bouts'].hist()
#not dropping it

X_train_num['B_win_by_TKO_Doctor_Stoppage'].value_counts(normalize=True)
X_train_num['R_win_by_TKO_Doctor_Stoppage'].value_counts(normalize=True)
X_train_num['B_win_by_TKO_Doctor_Stoppage'].hist()
#dropping this too
X_test_num.drop(['B_win_by_TKO_Doctor_Stoppage', 'R_win_by_TKO_Doctor_Stoppage'], axis=1, inplace=True)
X_train_num.drop(['B_win_by_TKO_Doctor_Stoppage', 'R_win_by_TKO_Doctor_Stoppage'], axis=1, inplace=True)

#now we will use some Random Forests to check the feature importance's 

rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42, max_leaf_nodes=16, class_weight="balanced_subsample")  

rnd_clf.fit(X_train_num, y_train)
for score, name in zip(rnd_clf.feature_importances_, X_train_num.columns):
    print(round(score, 3), name)
    
#lets see every feature that has lower than 0.007 feature importance
for score, name in zip(rnd_clf.feature_importances_, X_train_num.columns):
    if score < 0.008:
        print(round(score, 3), name)
    
#we choose to drop every feature that has both R and B, and some others as well
X_train_num.drop(['B_current_lose_streak', 'B_total_title_bouts', 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission','R_current_lose_streak', 'R_total_title_bouts', 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'no_of_rounds', 'total_title_bout_dif', 'lose_streak_dif'], axis=1, inplace=True)
X_test_num.drop(['B_current_lose_streak', 'B_total_title_bouts', 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission','R_current_lose_streak', 'R_total_title_bouts', 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'no_of_rounds', 'total_title_bout_dif', 'lose_streak_dif'], axis=1, inplace=True)
X_train_num.columns
#Lets check for outliers

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

plot_X_train_num_iqr = X_train_num.copy()

for column in plot_X_train_num_iqr.columns:
    mark_outliers_iqr(plot_X_train_num_iqr, column)



for column in X_train_num.columns:
    plot_binary_outliers(plot_X_train_num_iqr, column, column + '_outlier', False)
    
#cols to note with this criterion: 
cols_for_iqr = ['B_avg_TD_pct', 'B_Height_cms', 'B_Reach_cms', 'R_avg_TD_pct', 'R_age', 'B_age']
#we see that with the iqr method many of the columns have a lot of outliers that we can not exlude,lets try another approach
    
plot_X_train_num_Chauvenets = X_train_num.copy()

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

for column in X_train_num.columns:
    mark_outliers_chauvenet(plot_X_train_num_Chauvenets, column)


#we should consider tho that the Chauvenet's works best for data that follows a normal distribution.So if we configure some outliers based on this criterion we will do so if the distribution seems normal.
normal_dist_col = ['B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_Height_cms', 'B_Reach_cms','R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_Height_cms', 'R_Reach_cms', 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'Rank_dif']

for column in normal_dist_col:
    plot_binary_outliers(plot_X_train_num_Chauvenets, column, column + '_outlier', False)
    

#cols to note with this criterion: 
cols_for_chauv = ['B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct','longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'height_dif', 'reach_dif', 'Rank_dif']
#we didnt include some cols to note that we think that its not proper to change the values of them (eg total Takedown dif because of styles of fighters etc)
#Let's see them one by one
for column in cols_for_iqr:
    plot_binary_outliers(plot_X_train_num_iqr, column, column + '_outlier', False)

plot_X_train_num_iqr['B_avg_TD_pct_outlier'].sum()
#outliers are marked as True
#B_avg_TD_pct
plot_X_train_num_iqr[plot_X_train_num_iqr['B_avg_TD_pct_outlier']]['B_avg_TD_pct'].value_counts()
plot_X_train_num_iqr[plot_X_train_num_iqr['B_avg_TD_pct_outlier']]['B_avg_TD_pct'].min()
X_train_num[X_train_num['B_avg_TD_pct'] > 0.9]['B_avg_TD_pct'].replace(to_replace=[1, 0.9] ,value=0.89, inplace=True)
for i in X_train_num[X_train_num['B_avg_TD_pct'] > 0.9]['B_avg_TD_pct'].index:
    X_train_num['B_avg_TD_pct'][i] = 0.89 
    print(X_train_num['B_avg_TD_pct'][i])

X_train_num['B_avg_TD_pct'].max()

#B_Height_cms
plot_X_train_num_iqr[plot_X_train_num_iqr['B_Height_cms_outlier']]['B_Height_cms'].value_counts()
plot_X_train_num_iqr[~plot_X_train_num_iqr['B_Height_cms_outlier']]['B_Height_cms'].max()
plot_X_train_num_iqr[~plot_X_train_num_iqr['B_Height_cms_outlier']]['B_Height_cms'].min()
X_train_num['B_Height_cms'].max()
X_train_num['B_Height_cms'].min()
for index in plot_X_train_num_iqr[plot_X_train_num_iqr['B_Height_cms_outlier']]['B_Height_cms'].index:
    if X_train_num['B_Height_cms'][index] == 210.82:
        X_train_num['B_Height_cms'][index] = 203
    if X_train_num['B_Height_cms'][index] == 152.40:
        X_train_num['B_Height_cms'][index] = 155

#B_Reach_cms
plot_X_train_num_iqr[plot_X_train_num_iqr['B_Reach_cms_outlier']]['B_Reach_cms'].value_counts()
plot_X_train_num_iqr[~plot_X_train_num_iqr['B_Reach_cms_outlier']]['B_Reach_cms'].min()
for index in plot_X_train_num_iqr[plot_X_train_num_iqr['B_Reach_cms_outlier']]['B_Reach_cms'].index:
    if (X_train_num['B_Reach_cms'][index] == 149.86) or (X_train_num['B_Reach_cms'][index] == 147.32):
        X_train_num['B_Reach_cms'][index] = 152.5

X_train_num['B_Reach_cms'].min()

#R_avg_TD_pct
plot_X_train_num_iqr[plot_X_train_num_iqr['R_avg_TD_pct_outlier']]['R_avg_TD_pct'].min()
plot_X_train_num_iqr[~plot_X_train_num_iqr['R_avg_TD_pct_outlier']]['R_avg_TD_pct'].max()
for index in plot_X_train_num_iqr[plot_X_train_num_iqr['R_avg_TD_pct_outlier']]['R_avg_TD_pct'].index:
    if (X_train_num['R_avg_TD_pct'][index] > 0.833333333):
        X_train_num['R_avg_TD_pct'][index] = 0.83

X_train_num['R_avg_TD_pct'].max()

#R_age
plot_X_train_num_iqr[plot_X_train_num_iqr['R_age_outlier']]['R_age'].min()
plot_X_train_num_iqr[~plot_X_train_num_iqr['R_age_outlier']]['R_age'].max()
for index in plot_X_train_num_iqr[plot_X_train_num_iqr['R_age_outlier']]['R_age'].index:
    if (X_train_num['R_age'][index] > 43):
        X_train_num['R_age'][index] = 42

X_train_num['R_age'].max()

#B_age
plot_X_train_num_iqr[plot_X_train_num_iqr['B_age_outlier']]['B_age'].value_counts()
for index in plot_X_train_num_iqr[plot_X_train_num_iqr['B_age_outlier']]['B_age'].index:
    if (X_train_num['B_age'][index] > 39.5):
        X_train_num['B_age'][index] = 39
    if (X_train_num['B_age'][index] < 19.5):
        X_train_num['B_age'][index] = 20

X_train_num['B_Reach_cms'].min()

#Now for the other criterion

for column in cols_for_chauv:
    plot_binary_outliers(plot_X_train_num_Chauvenets, column, column + '_outlier', False)

#i will automate this (dropping all the outliers for the remaining cols)
for column in cols_for_chauv:
    for index in plot_X_train_num_Chauvenets[plot_X_train_num_Chauvenets[column+'_outlier']][column].index:
        if (X_train_num[column][index] > plot_X_train_num_Chauvenets[~plot_X_train_num_Chauvenets[column+'_outlier']][column].max()):
           X_train_num[column][index] = plot_X_train_num_Chauvenets[~plot_X_train_num_iqr[column+'_outlier']][column].max()
        if (X_train_num[column][index] < plot_X_train_num_Chauvenets[~plot_X_train_num_Chauvenets[column+'_outlier']][column].min()):
            X_train_num[column][index] = plot_X_train_num_Chauvenets[~plot_X_train_num_Chauvenets[column+'_outlier']][column].min()
       
X_train_num.columns


for column in X_train_num.columns:
    X_train_num[column].hist(legend=True)
    plt.show()

heavy_tail_num_cols_B = ['B_current_win_streak', 'B_avg_SIG_STR_landed', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_losses','B_total_rounds_fought', 'B_wins']

for column in heavy_tail_num_cols_B:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2.hist(np.sqrt(X_train_num[column]))
    ax1.hist(X_train_num[column])
    ax3.hist(np.log10(X_train_num[column] + 0.00000000000001)) #to avoid log(0) = inf
    ax1.set_title(column)
    ax2.set_title('sqrt' + column)
    ax3.set_title('log' + column)
    plt.show()
    
#we see that the sqrt approach makes the histogram effectively more bell curved for the next cols
cols_for_sqrt = ['B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_total_rounds_fought', 'B_wins', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_longest_win_streak', 'R_total_rounds_fought', 'R_wins']

for col in cols_for_sqrt:
    X_train_num['sqrt_'+col] = np.sqrt(X_train_num[col])

X_train_num.drop(cols_for_sqrt, axis=1, inplace=True)
X_train_num 

for col in cols_for_sqrt:
    X_test_num['sqrt_'+col] = np.sqrt(X_test_num[col])

X_test_num.drop(cols_for_sqrt, axis=1, inplace=True)
X_train_num.columns

remaining_cols_with_tail = ['B_current_win_streak', 'B_avg_SUB_ATT', 'B_losses', 'R_current_win_streak', 'R_avg_SUB_ATT', 'R_losses']
for column in remaining_cols_with_tail:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.hist(X_train_num[column])
    ax2.hist(np.cbrt(X_train_num[column]))
    ax3.hist(np.arcsin(np.sqrt(X_train_num[column])))
    ax1.set_title(column)
    ax2.set_title('cube_root_'+column)
    ax3.set_title('arcsinsqrt_'+column)
    plt.show()

#we will use the cube root of the remaining cols
for col in remaining_cols_with_tail:
    X_train_num['cbrt_'+col] = np.sqrt(X_train_num[col])

X_train_num.drop(remaining_cols_with_tail, axis=1, inplace=True)

for col in remaining_cols_with_tail:
    X_test_num['cbrt_'+col] = np.sqrt(X_test_num[col])

X_test_num.drop(remaining_cols_with_tail, axis=1, inplace=True)

#Standar scale the dataset

std_scaler = StandardScaler()
X_train_num_scaled_values = std_scaler.fit_transform(X_train_num)
std_scaler.get_feature_names_out()
X_train_num_scaled = pd.DataFrame(X_train_num_scaled_values, columns=std_scaler.get_feature_names_out(), index=X_train_num.index)

X_test_num_scaled_values = std_scaler.transform(X_test_num)
X_test_num_scaled = pd.DataFrame(X_test_num_scaled_values, columns=std_scaler.get_feature_names_out(), index=X_test_num.index)


for column in X_train_num_scaled.columns:
    X_train_num_scaled[column].hist(legend=True)
    plt.show()
'''
X_train_num_scaled.to_csv('X_train_num_scaled.csv', index=False)
X_test_num_scaled.to_csv('X_test_num_scaled.csv', index=False)
'''

pca = PCA(n_components=0.95)
X_train_num_scaled_pca_values = pca.fit_transform(X_train_num_scaled)
pca.n_components_
len(X_train_num_scaled.columns)
#We see that we can have 95% explained variance ratio with roughly half of the features.

X_train_num_scaled_pca = pd.DataFrame(X_train_num_scaled_pca_values, columns=pca.get_feature_names_out(), index=X_train_num.index)

X_test_num_scaled_pca_values = pca.transform(X_test_num_scaled)
X_test_num_scaled_pca = pd.DataFrame(X_test_num_scaled_pca_values, columns=pca.get_feature_names_out(), index=X_test_num.index)

for column in X_train_num_scaled_pca.columns:
    X_train_num_scaled_pca[column].hist(legend=True)
    plt.show()

#we see that the pca features are much bell shaped like

#We will combine the cat and num training values

X_train_concat = pd.concat([X_train_num_scaled_pca, X_train_cat_en], axis=1)
X_test_concat = pd.concat([X_test_num_scaled_pca, X_test_cat_en], axis=1)
#Saving of the datasets
'''
X_train_concat.to_csv('X_train_concat.csv', index=False)
X_test_concat.to_csv('X_test_concat.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
datasets that are stratified based on weight class
'''
'''
X_train_concat.to_csv('X_train_concat_branch.csv', index=False)
X_test_concat.to_csv('X_test_concat_branch.csv', index=False)
y_train.to_csv('y_train_branch.csv', index=False)
y_test.to_csv('y_test_branch.csv', index=False)
datasets that are stratified based on y

X_train_num_scaled.to_csv('X_train_num_scaled_branch.csv', index=False)
X_test_num_scaled.to_csv('X_test_num_scaled_branch.csv', index=False)
'''

X_train_without_pca_full_same_weight_class_ratios= pd.read_csv('./X_train_num_scaled.csv', dtype=np.float64)
for col in X_train_without_pca_full_same_weight_class_ratios.columns:
    print(col)
col_names_full = []
for column in X_train_without_pca_full_same_weight_class_ratios.columns[:26]:
    col_names_full.append(column)
col_names_full


for col in X_train_without_pca_full_same_weight_class_ratios.columns[26:]:
    col_names_full.append(col[5:])
    
col_names_full
dif_cols = ['win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'Rank_dif']
sqrt_cols = ['B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_total_rounds_fought', 'B_wins', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_longest_win_streak', 'R_total_rounds_fought', 'R_wins']
cbrt_cols = ['B_current_win_streak', 'B_avg_SUB_ATT', 'B_losses', 'R_current_win_streak', 'R_avg_SUB_ATT', 'R_losses']
sq_cb_cols = sqrt_cols + cbrt_cols
rest_cols = ['B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_win_by_Decision_Split', 'B_Height_cms', 'B_Reach_cms', 'R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_win_by_Decision_Split', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'B_age']




imputer_dif = SimpleImputer(strategy="constant", fill_value=0)
imputer_sqrt = SimpleImputer(strategy="mean")
imputer_rest = SimpleImputer(strategy="mean")




sqrt_transformer = FunctionTransformer(np.sqrt, inverse_func=np.square, feature_names_out= 'one-to-one')


diff_pipeline = Pipeline([
    ("impute", imputer_dif)
])

sqrt_pipeline = Pipeline([
    ("impute", imputer_sqrt),
    ("sqrt", sqrt_transformer)
])

rest_pipeline = Pipeline([
    ("impute", imputer_rest)
])

Column_trans = ColumnTransformer([
    ("dif", diff_pipeline, dif_cols),
    ("sqrt", sqrt_pipeline, sq_cb_cols),
    ("rest", rest_pipeline, rest_cols)
], 
verbose_feature_names_out=False)

Column_trans.fit(df5[col_names_full])

with open('Colum_transf.pkl','wb') as f:
    pickle.dump(Column_trans, f)
    
standar_pca_pipeline = Pipeline([
    ("StandarScaler", std_scaler),
    ("PCA", pca)
])

with open('standar_pca_pipe.pkl','wb') as f:
    pickle.dump(standar_pca_pipeline, f)



R_weight_classes = ["R_Women's Featherweight_rank", "R_Women's Strawweight_rank", "R_Women's Bantamweight_rank", 'R_Heavyweight_rank', 'R_Light Heavyweight_rank', 'R_Middleweight_rank', 'R_Welterweight_rank', 'R_Lightweight_rank', 'R_Featherweight_rank', 'R_Bantamweight_rank', 'R_Flyweight_rank']
B_weight_classes = ["B_Women's Featherweight_rank", "B_Women's Strawweight_rank", "B_Women's Bantamweight_rank", 'B_Heavyweight_rank', 'B_Light Heavyweight_rank', 'B_Middleweight_rank', 'B_Welterweight_rank', 'B_Lightweight_rank', 'B_Featherweight_rank', 'B_Bantamweight_rank', 'B_Flyweight_rank']
mixed_dif = ['loss_dif', 'ko_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'total_round_dif', 'sub_dif']
mixed_B = ['B_losses', 'B_win_by_KO/TKO', 'B_Height_cms', 'B_Reach_cms', 'B_age', 'B_avg_SIG_STR_landed', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_current_win_streak', 'B_longest_win_streak', 'B_wins', 'B_total_rounds_fought', 'B_win_by_Submission']
mixed_R = ['R_losses', 'R_win_by_KO/TKO', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'R_avg_SIG_STR_landed', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_current_win_streak', 'R_longest_win_streak', 'R_wins', 'R_total_rounds_fought', 'R_win_by_Submission']
cols_in_order = ['B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_win_by_Decision_Split', 'B_Height_cms', 'B_Reach_cms', 'R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_win_by_Decision_Split', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'B_age', 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'Rank_dif', 'sqrt_B_avg_SIG_STR_landed', 'sqrt_B_avg_TD_landed', 'sqrt_B_longest_win_streak', 'sqrt_B_total_rounds_fought', 'sqrt_B_wins', 'sqrt_R_avg_SIG_STR_landed', 'sqrt_R_avg_TD_landed', 'sqrt_R_longest_win_streak', 'sqrt_R_total_rounds_fought', 'sqrt_R_wins', 'cbrt_B_current_win_streak', 'cbrt_B_avg_SUB_ATT', 'cbrt_B_losses', 'cbrt_R_current_win_streak', 'cbrt_R_avg_SUB_ATT', 'cbrt_R_losses']
sqrt_cols = ['B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_total_rounds_fought', 'B_wins', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_longest_win_streak', 'R_total_rounds_fought', 'R_wins']
cbrt_cols = ['B_current_win_streak', 'B_avg_SUB_ATT', 'B_losses', 'R_current_win_streak', 'R_avg_SUB_ATT', 'R_losses']

def preprocessing(New_data):
    New_data.reset_index(drop=True, inplace=True)
    R_Series = []
    B_Series = []
    for j in range(New_data.shape[0]):
        R_Series.append(16)
        B_Series.append(16)
    
    for j in range(New_data.shape[0]):
        for column in R_weight_classes:
            if ~New_data.isna().iloc[j][column]:
                R_Series[j] = New_data.iloc[j][column]
    
    for j in range(New_data.shape[0]):
        for column in B_weight_classes:
            if ~New_data.isna().iloc[j][column]:
                B_Series[j] = New_data.iloc[j][column]

    New_data['R_rank'] = R_Series
    New_data['B_rank'] = B_Series
    New_data['B_rank'].replace(0, 16, inplace=True)
    New_data['R_rank'].replace(0, 16, inplace=True)
    
    New_data['Rank_dif'] = New_data['B_rank'] - New_data['R_rank']
    New_data.drop(["R_rank" ,"B_rank"], axis=1, inplace=True)
    
    
    for j in range(len(mixed_dif)):
        for i in range(New_data.shape[0]):
            if ~(New_data[mixed_B[j]] - New_data[mixed_R[j]] == New_data[mixed_dif[j]])[i]:
                New_data[mixed_dif[j]][i] = New_data[mixed_B[j]][i] - New_data[mixed_R[j]][i] 
    
    New_data_prepared = Column_trans.transform(New_data)
    New_data_prepared_df = pd.DataFrame(New_data_prepared, columns=Column_trans.get_feature_names_out(), index=New_data.index)
    
   
    for col in sqrt_cols:
        New_data_prepared_df['sqrt_'+col] = New_data_prepared_df[col]
    
    
    
    New_data_prepared_df.drop(['B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_total_rounds_fought', 'B_wins', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_longest_win_streak', 'R_total_rounds_fought', 'R_wins'], axis=1, inplace=True)
    
    for col in cbrt_cols:
        New_data_prepared_df['cbrt_'+col] = New_data_prepared_df[col]
        
    New_data_prepared_df.drop(['B_current_win_streak', 'B_avg_SUB_ATT', 'B_losses', 'R_current_win_streak', 'R_avg_SUB_ATT', 'R_losses'], axis=1, inplace=True)
    
    
    
    New_data_prepared_df = New_data_prepared_df[cols_in_order]
    X_final = standar_pca_pipeline.transform(New_data_prepared_df)
    X_final_df = pd.DataFrame(X_final, columns=standar_pca_pipeline.get_feature_names_out(), index=New_data.index)
    return X_final_df
    
    
j = 0
k = 0
for i in range(df1.shape[0]):
    if (preprocessing(df1[i:(i+1)]).shape[1] == 23):
        j = j +1
    else:
        k = k + 1
print(j)
#4896 so every instance would have 23 pca features just as we want it
print(k)
#0


prepro_data = preprocessing(df1)



essential_cols_1 = sq_cb_cols + rest_cols + mixed_B + mixed_R + mixed_dif + R_weight_classes + B_weight_classes
essential_cols = list(set(essential_cols_1))
essential_cols.append("R_fighter")
essential_cols.append("B_fighter")
df_ess = df[essential_cols]
df_ess
df_R_1 = df_ess.copy()
df_R_1['B_fighter'].unique()

fighters_unique_in_red_not_in_blue = []
for fighter in df['R_fighter'].unique():
    if fighter not in df['B_fighter'].unique():
        fighters_unique_in_red_not_in_blue.append(fighter)

len(fighters_unique_in_red_not_in_blue)
fighters_unique_in_blue = []
for fighter in df['B_fighter'].unique():
    fighters_unique_in_blue.append(fighter)
    
fighters_unique_in_blue

unique_fighters_names = fighters_unique_in_blue + fighters_unique_in_red_not_in_blue



ids_of_reds = []
names = []

for index in df_R_1.index:
    if ((df_R_1['R_fighter'][index] in fighters_unique_in_red_not_in_blue) and (df_R_1['R_fighter'][index] not in names)):
        names.append(df_R_1['R_fighter'][index])
        ids_of_reds.append(index)


len(ids_of_reds)
len(names)

R_cols = []

for col in df_R_1.columns:
    if col.startswith('R'):
        R_cols.append(col)
        
B_cols = []

for col in df_R_1.columns:
    if col.startswith('B'):
        B_cols.append(col)
     
len(R_cols)   
df_R = df_R_1.iloc[ids_of_reds][R_cols].reset_index(drop=True)
df_R.columns

columns_without_R = [col.replace('R_', '', 1) for col in df_R.columns]
renaming = {}
for i in range(df_R.shape[1]):
    renaming[df_R.columns[i]] = columns_without_R[i]

renaming

df_R = df_R.rename(columns=renaming)
df_R

df_B_1 = df_ess.copy()

ids_of_blues = []
names_B = []

for index in df_B_1.index:
    if ((df_B_1['B_fighter'][index] in fighters_unique_in_blue) and (df_B_1['B_fighter'][index] not in names_B)):
        names_B.append(df_B_1['B_fighter'][index])
        ids_of_blues.append(index)


len(ids_of_blues)
len(names_B)

B_cols = []

for col in df_B_1.columns:
    if col.startswith('B'):B_cols
        B_cols.append(col)
        

len(B_cols)   
df_B = df_B_1.iloc[ids_of_blues][B_cols].reset_index(drop=True)
df_B.columns

columns_without_B = [col.replace('B_', '', 1) for col in df_B.columns]
renaming_B = {}
for i in range(df_B.shape[1]):
    renaming_B[df_B.columns[i]] = columns_without_B[i]

renaming_B

df_B = df_B.rename(columns=renaming_B)
df_B

unique_fighters_stats = pd.concat([df_R, df_B], ignore_index=True) 

lower_case = unique_fighters_stats['fighter'].map(lambda x: x.lower() if isinstance(x,str) else x)
re.sub(r'[^\w]', ' ', s)
remove_symbols = lower_case.map(lambda x: re.sub(r'[^\w]', ' ', x))
remove_spaces = remove_symbols.str.replace(" ", "")

unique_fighters_stats['fighter'] = remove_spaces
unique_fighters_stats

#unique_fighters_stats.to_csv('unique_fighters_stats.csv', index=False)
unique_fighters_stats = pd.read_csv("./unique_fighters_stats.csv")

uniqueList = []
duplicateList = []
 
for i in list(unique_fighters_stats['fighter']):
    if i not in uniqueList:
        uniqueList.append(i)
    elif i not in duplicateList:
        duplicateList.append(i)

duplicateList


for dupl_fight in duplicateList:
    print(len(unique_fighters_stats[unique_fighters_stats['fighter'] == dupl_fight]))
    
#somehow there are 7 dublicates

list_of_dub_idx = []

for fighter in duplicateList:
    list_of_dub_idx.append(unique_fighters_stats[unique_fighters_stats['fighter'] == fighter].index[0])
    
list_of_dub_idx

unique_fighters_stats_final = unique_fighters_stats.drop(list_of_dub_idx, axis=0)
#unique_fighters_stats_final.to_csv('unique_fighters_stats_final.csv', index=False)

uniqueList_final = []
duplicateList_final = []
 
for i in list(unique_fighters_stats_final['fighter']):
    if i not in uniqueList_final:
        uniqueList_final.append(i)
    elif i not in duplicateList_final:
        duplicateList_final.append(i)

duplicateList_final
#Solved

ids_of_reds_final = [x for x in ids_of_reds if (x not in list_of_dub_idx)]
ids_of_blues_final = [x for x in ids_of_blues if (x not in list_of_dub_idx)]

for val in df['R_Weight_lbs'].unique():
    print(val)

weights = [115, 125, 135, 145, 155, 170, 185, 205, 265]
df_unique_red = df.iloc[ids_of_reds_final]
df_unique_red = df_unique_red.reset_index(drop=True)
df_unique_red.loc[(df_unique_red['R_Weight_lbs'] > 206), 'R_Weight_lbs']  = 265
df_unique_blue = df.iloc[ids_of_blues_final]
df_unique_blue = df_unique_blue.reset_index(drop=True)
df_unique_blue.loc[(df_unique_blue['B_Weight_lbs'] > 206), 'B_Weight_lbs']  = 265
list(df_unique_red[df_unique_red['R_Weight_lbs'] == 115]['R_fighter'])
dict_of_weights = {115 : [], 125: [], 135: [], 145: [], 155: [], 170: [], 185: [], 205: [], 265: []}
for weight in weights:
    for fighter in list(df_unique_red[df_unique_red['R_Weight_lbs'] == weight]['R_fighter']):
        dict_of_weights[weight].append(fighter)

for weight in weights:
    for fighter in list(df_unique_blue[df_unique_blue['B_Weight_lbs'] == weight]['B_fighter']):
        dict_of_weights[weight].append(fighter)        

dict_of_weights
dict_copy = dict_of_weights
dict_of_weights_final_1 = {115 : [], 125: [], 135: [], 145: [], 155: [], 170: [], 185: [], 205: [], 265: []}
for weight in dict_copy:
   dict_of_weights_final_1[weight].append([re.sub(r'[^\w]', ' ', x.lower()).replace(" ", "") for x in dict_of_weights[weight]])

for weight in dict_of_weights_final_1:
    dict_of_weights_final_1[weight] = dict_of_weights_final_1[weight][0]

dict_of_weights_final_1

check_num = 0
for weight in dict_of_weights_final_1:
    check_num = check_num + len(dict_of_weights_final_1[weight])

check_num



uniqueList = []
duplicateList = []
 
for weight in dict_of_weights_final_1:
    for i in dict_of_weights_final_1[weight]:
        if i not in uniqueList:
            uniqueList.append(i)
        elif i not in duplicateList:
            duplicateList.append(i)

duplicateList




def X_from_names(red_corner_name, blue_corner_name):

    red_name_adj = re.sub(r'[^\w]', ' ', red_corner_name.lower()).replace(" ", "")
    blue_name_adj = re.sub(r'[^\w]', ' ', blue_corner_name.lower()).replace(" ", "")

    for pos_weight in dict_of_weights_final_1:
        for name in dict_of_weights_final_1[pos_weight]:
            if name == red_name_adj:
                red_weight = pos_weight  
    
    if blue_name_adj not in dict_of_weights_final_1[red_weight]:
        return print("The fighters are not in the same weight class") 

    if ((red_name_adj not in list(unique_fighters_stats['fighter'])) and (blue_name_adj not in list(unique_fighters_stats['fighter']))):
        return print("Both given names are not in the list")
    if red_name_adj not in list(unique_fighters_stats['fighter']):
        return print("The name given in the red corner is not in the list")
    if blue_name_adj not in list(unique_fighters_stats['fighter']):
        return print("The name given in the blue corner is not in the list")  
    
    if red_name_adj == blue_name_adj:
        return print("You gave the same name in both corners") 
    
    
    
    if (red_name_adj in list(unique_fighters_stats['fighter'])):
        R_frame = unique_fighters_stats[unique_fighters_stats['fighter'] == red_name_adj]
        R_frame_adj = R_frame.add_prefix('R_')
        R_frame_adj = R_frame_adj.reset_index(drop=True)
   
       
    if (blue_name_adj in list(unique_fighters_stats['fighter'])):
        B_frame = unique_fighters_stats[unique_fighters_stats['fighter'] == blue_name_adj]
        B_frame_adj = B_frame.add_prefix('B_')
        B_frame_adj = B_frame_adj.reset_index(drop=True)

        
    X_without_pre = pd.merge(R_frame_adj, B_frame_adj, left_index=True, right_index=True)
    return X_without_pre
    
    
X_from_names('caludiagadelha', 'jessicaandrade')

