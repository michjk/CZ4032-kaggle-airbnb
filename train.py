import numpy as np 
import xgboost as xgb 
import pandas as pd
import time
from data_manager.data_preprocessor import load_preprocessed_data

#dataset path
dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"

x_train, y_train, x_test = load_preprocessed_data(train_path, test_path)

#train
clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.09, n_estimators=125, objective="multi:softprob", subsample=0.5, colsample_bytree=0.5, seed=0)
t = time.time()
clf.fit(x_train, y_train)
print("Training time %lf"%(time.time()-t))
y_pred = clf.predict_proba(x_test)

id_out = []
label_out = []
for i in range(len(x_test)):
    user_id = id_test[i]
    id_out += [user_id]*5
    sorted_args_pos = np.argsort(y_pred[i])[::-1]
    label_name_pred = label_encoder.inverse_transform(sorted_args_pos)
    label_out += label_name_pred[:5].tolist()

submission = pd.DataFrame(np.column_stack((id_out, label_out)), columns=[id_name, country_name])
submission.to_csv("sub.csv", index=False)
