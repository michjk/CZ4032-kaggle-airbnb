import pandas as pd
from data_constants import *

def create_data_NDF_and_non_DBF(train_path, new_train_path):
    df_train = pd.read_csv(train_path)
    df_train[country_destination_name] = df_train[country_destination_name].apply(lambda x: "non-NDF" if x != "NDF" else x)
    df_train.to_csv(new_train_path)
