import xgboost as xgb 
from data_manager.data_preprocessor import load_preprocessed_data

def initializeDefaultParams():
    params = {}
    params['objective'] = "multi:softprob"
    params['num_class'] = 2

    return params

def initializeDefaultParamsGPU():
    params = initializeDefaultParams()
    params['tree_method'] = 'gpu_exact'
    return params

dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2_NDF_vs_non_NDF.csv"
test_path = dataset_folder_path + "test_users.csv"

num_boost_round_default = 250
nfold = 5
seed = 0
metrics = ['mlogloss']

x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
train_dmat = xgb.DMatrix(x_train, y_train)

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(2,12)
    for min_child_weight in range(2,12)
]

min_mlogloss = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

    params = initializeDefaultParamsGPU()

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_result = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=25
    )

    mean_mlogloss = cv_result['test-mlogloss-mean'].min()
    boost_rounds = cv_result['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (max_depth,min_child_weight)

print(best_params)