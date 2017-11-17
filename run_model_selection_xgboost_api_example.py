import xgboost as xgb


# In[ ]:


from data_manager.data_preprocessor import load_preprocessed_data
import time


# In[ ]:


def initializeDefaultParams():
    params = {}
    params['objective'] = "multi:softprob"
    params['num_class'] = 12
    params['nthread'] = -1

    return params

def initializeDefaultParamsGPU():
    params = initializeDefaultParams()
    params['tree_method'] = 'gpu_hist'
    return params


# In[ ]:


dataset_folder_path = "../dataset/"
train_path = dataset_folder_path + "train_users_2.csv"
test_path = dataset_folder_path + "test_users.csv"


# In[ ]:


num_boost_round_default = 999
early_stopping_round_default = 10
nfold = 5
seed = 0
metrics = ['mlogloss']


# In[ ]:


x_train, y_train, x_test, id_test, label_encoder = load_preprocessed_data(train_path, test_path)
train_dmat = xgb.DMatrix(x_train, y_train)


# In[ ]:


gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(2,8)
    for min_child_weight in range(2,8)
]


# In[ ]:


min_mlogloss = float("Inf")
best_params = None
import time
t = time.time()
#search for max depth and min_child_weight
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
        early_stopping_rounds=early_stopping_round_default,
        verbose_eval = True
        
    )

    mean_mlogloss = cv_result['test-mlogloss-mean'].min()
    boost_rounds = cv_result['test-mlogloss-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (max_depth,min_child_weight)

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))
print("Time: {}".format(time.time()-t))
max_depth = best_params[0]
min_child_weight = best_params[1]


# In[ ]:


gridsearch_params = [
    (subsample, colsample_bytree)
    for subsample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    for colsample_bytree in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
]

#search col and row
min_mlogloss= float("Inf")
best_params = None
for (subsample, colsample_bytree) in gridsearch_params:
    print("CV with subsample={} colsample_bytree={}".format(subsample, colsample_bytree))

    params = initializeDefaultParamsGPU()

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample_bytree
    
    cv_results = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=early_stopping_round_default
    )

    # Update best score
    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = eta

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))
subsample = best_params[0]
colsample_bytree = best_params[1]


# In[ ]:


#search eta
list_eta = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
min_mlogloss= float("Inf")
best_params = None
for eta in list_eta:
    print("CV with eta={}".format(eta))

    params = initializeDefaultParamsGPU()

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample_bytree
    params['eta'] = eta

    cv_results = xgb.cv(
        params,
        train_dmat,
        num_boost_round=num_boost_round_default,
        seed=seed,
        nfold=nfold,
        metrics=metrics,
        early_stopping_rounds=25
    )

    # Update best score
    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = eta

print("Best params: {}, mlogloss: {}".format(best_params, min_mlogloss))
eta = best_params
