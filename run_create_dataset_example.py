from data_manager.data_generator import create_data_exclude_NDF

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
new_train_path = dataset_folder_path + "train_users_exclude_NDF.csv"

create_data_exclude_NDF(train_path, new_train_path)