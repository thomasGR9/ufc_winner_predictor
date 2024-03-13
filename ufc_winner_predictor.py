import pandas as pd 
df = pd.read_csv('../../datasets/ufc-master.csv')
for column in df.columns:
    print(column)

df['finish']


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




~df2.iloc[:, 68:79].isna().iloc[0]
R_Series = []
for j in range(df2.shape[0]):
    R_Series.append(16)

R_Series

for j in range(df2.shape[0]):
    for column in df2.iloc[:, 68:79].columns:
        if ~df2.iloc[:, 68:79].isna().iloc[j][column]:
            R_Series[j] = df2.iloc[:, 68:79].iloc[j][column]
   
R_Series        
            
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
#so when the diff is not B - R s R - B anid not a mistake
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
import numpy as np
df5.columns
#the columns we will use to predict the winner have relative importance to the gender and weight of the fighters.To avoid biases we will stratify according to weight_class witch combines the two factors by specifying womans and mans weight class
df5['weight_class'].value_counts() 
#so we can drop the gender and weight columns because their information is in the weight class column
df5.drop(['gender', 'B_Weight_lbs', 'R_Weight_lbs'], axis=1, inplace=True)
y = df5['Winner']
x = df5.drop(['Winner'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=x['weight_class'], test_size=0.2, shuffle=True)
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

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
train_cat_en = cat_encoder.fit_transform(X_train_cat)
X_train_cat_en = pd.DataFrame(train_cat_en.toarray(), columns=cat_encoder.get_feature_names_out(), index=X_train_cat.index )
X_train_cat_en
cat_encoder.transform(X_test_cat).toarray()
X_test_cat_en = pd.DataFrame(cat_encoder.transform(X_test_cat).toarray(), columns=cat_encoder.get_feature_names_out(), index=X_test_cat.index )
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
cat_encoder_y = OneHotEncoder()
train_cat_en_y = cat_encoder_y.fit_transform(pd.DataFrame(y_train, columns = ['Winner'], index = y_train.index))
y_train_en = pd.DataFrame(train_cat_en_y.toarray(), columns=cat_encoder_y.get_feature_names_out(), index=y_train.index )
test_cat_en_y = cat_encoder_y.fit_transform(pd.DataFrame(y_test, columns = ['Winner'], index = y_test.index))
y_test_en = pd.DataFrame(test_cat_en_y.toarray(), columns=cat_encoder_y.get_feature_names_out(), index=y_test.index)

