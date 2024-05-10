import pickle
import tensorflow as tf
from keras.models import load_model
import pandas as pd

with open("top_mod_RFC.pkl", "rb") as f:
    top_mod_RFC = pickle.load(f)

with open("top_mod_XGB.pkl", "rb") as f:
    top_mod_XGB = pickle.load(f)
    
with open("bayes.pkl", "rb") as f:
    bayes = pickle.load(f)
    
with open("meta_learner.pkl", "rb") as f:
    meta_learner = pickle.load(f)
    
top_mod_NN = tf.keras.models.load_model('./61acc.h5')

with open("Colum_transf.pkl", "rb") as f:
    Column_trans = pickle.load(f)

with open("standar_pca_pipe.pkl", "rb") as f:
    standar_pca_pipeline = pickle.load(f)

R_weight_classes = ["R_Women's Featherweight_rank", "R_Women's Strawweight_rank", "R_Women's Bantamweight_rank", 'R_Heavyweight_rank', 'R_Light Heavyweight_rank', 'R_Middleweight_rank', 'R_Welterweight_rank', 'R_Lightweight_rank', 'R_Featherweight_rank', 'R_Bantamweight_rank', 'R_Flyweight_rank']
B_weight_classes = ["B_Women's Featherweight_rank", "B_Women's Strawweight_rank", "B_Women's Bantamweight_rank", 'B_Heavyweight_rank', 'B_Light Heavyweight_rank', 'B_Middleweight_rank', 'B_Welterweight_rank', 'B_Lightweight_rank', 'B_Featherweight_rank', 'B_Bantamweight_rank', 'B_Flyweight_rank']
mixed_dif = ['loss_dif', 'ko_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'total_round_dif', 'sub_dif']
mixed_B = ['B_losses', 'B_win_by_KO/TKO', 'B_Height_cms', 'B_Reach_cms', 'B_age', 'B_avg_SIG_STR_landed', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_current_win_streak', 'B_longest_win_streak', 'B_wins', 'B_total_rounds_fought', 'B_win_by_Submission']
mixed_R = ['R_losses', 'R_win_by_KO/TKO', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'R_avg_SIG_STR_landed', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_current_win_streak', 'R_longest_win_streak', 'R_wins', 'R_total_rounds_fought', 'R_win_by_Submission']
cols_in_order = ['B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_win_by_Decision_Split', 'B_Height_cms', 'B_Reach_cms', 'R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_win_by_Decision_Split', 'R_Height_cms', 'R_Reach_cms', 'R_age', 'B_age', 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif', 'total_round_dif', 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif', 'avg_td_dif', 'Rank_dif', 'sqrt_B_avg_SIG_STR_landed', 'sqrt_B_avg_TD_landed', 'sqrt_B_longest_win_streak', 'sqrt_B_total_rounds_fought', 'sqrt_B_wins', 'sqrt_R_avg_SIG_STR_landed', 'sqrt_R_avg_TD_landed', 'sqrt_R_longest_win_streak', 'sqrt_R_total_rounds_fought', 'sqrt_R_wins', 'cbrt_B_current_win_streak', 'cbrt_B_avg_SUB_ATT', 'cbrt_B_losses', 'cbrt_R_current_win_streak', 'cbrt_R_avg_SUB_ATT', 'cbrt_R_losses']
sqrt_cols = ['B_avg_SIG_STR_landed', 'B_avg_TD_landed', 'B_longest_win_streak', 'B_total_rounds_fought', 'B_wins', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_longest_win_streak', 'R_total_rounds_fought', 'R_wins']
cbrt_cols = ['B_current_win_streak', 'B_avg_SUB_ATT', 'B_losses', 'R_current_win_streak', 'R_avg_SUB_ATT', 'R_losses']
Threshold_prec_dict = {
 0.5: 0.584,
 0.51: 0.588,
 0.52: 0.593,
 0.53: 0.595,
 0.54: 0.59,
 0.55: 0.601,
 0.56: 0.601,
 0.57: 0.599,
 0.58: 0.591,
 0.59: 0.592,
 0.6: 0.595,
 0.61: 0.597,
 0.62: 0.601,
 0.63: 0.611,
 0.64: 0.613,
 0.65: 0.606,
 0.66: 0.602,
 0.67: 0.614,
 0.68: 0.62,
 0.69: 0.621,
 0.7: 0.625,
 0.71: 0.627,
 0.72: 0.639,
 0.73: 0.655,
 0.74: 0.661,
 0.75: 0.688,
 0.76: 0.705,
 0.77: 0.716,
 0.78: 0.708,
 0.79: 0.705,
 0.8: 0.711,
 0.81: 0.741,
 0.82: 0.711,
 0.83: 0.741,
 0.84: 0.836,
 0.85: 0.875,
 0.86: 0.868,
 0.87: 0.885
}


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
    


def meta_learner_input(test_set_x):
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


def results(X_New_Instance):
    X_New_Instance_preprocessed_df =  preprocessing(New_data = X_New_Instance)
    meta_learner_input_df = meta_learner_input(test_set_x = X_New_Instance_preprocessed_df)
    preds = meta_learner.predict_proba(meta_learner_input_df)
    winner_blue_corner_proba = preds[:, 1]
    for i in range(meta_learner_input_df.shape[0]):
        if (winner_blue_corner_proba[i] > 0.5):
            winner = "Blue corner"
            winner_probability = round(winner_blue_corner_proba[i], 2)
        elif (winner_blue_corner_proba[i] < 0.5):
            winner = "Red corner"
            winner_probability = round((1 - winner_blue_corner_proba[i]), 2)
        elif (winner_blue_corner_proba[i] == 0.5):
            print("The predicted probability is equal for both corners")
        print(f"For the input number {i} \nThe winner is the fighter in the {winner} with predicted probability = {winner_probability}\nWith this predicted probability as threshold the model achieved {Threshold_prec_dict[winner_probability]} precision, in the test set")
    
        
