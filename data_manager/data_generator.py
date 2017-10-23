import pandas as pd
import numpy as np
from data_constants import *

def create_data_NDF_and_non_DBF(train_path, new_train_path):
    df_train = pd.read_csv(train_path)
    df_train[country_destination_name] = df_train[country_destination_name].apply(lambda x: "non-NDF" if x != "NDF" else x)
    df_train.to_csv(new_train_path)

def create_kaggle_submission_csv(y_pred, id_test, label_encoder, file_path, num_result_per_user = 5):
    id_out = []
    label_out = []
    for i in range(len(y_pred)):
        user_id = id_test[i]
        id_out += [user_id]*num_result_per_user
        sorted_args_pos = np.argsort(y_pred[i])[::-1]
        label_name_pred = label_encoder.inverse_transform(sorted_args_pos)
        label_out += label_name_pred[:num_result_per_user].tolist()

    submission = pd.DataFrame(np.column_stack((id_out, label_out)), columns=[id_name, country_name])
    submission.to_csv(file_path, index=False)
