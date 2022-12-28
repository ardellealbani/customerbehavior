"""
This is a boilerplate pipeline 'data_prepocessing'
generated using Kedro 0.18.3
"""

# Import the packages and libraries needed for this project
import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from sklearn.utils import resample
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool

def preparation(dataset_raw: pd.DataFrame, label: str):
    # Remove Column
    dataset = dataset_raw.drop(['index', 'month_id', 'movement', 'fea_3', 'fea_11', 'fea_49', 'fea_118'], axis=1)
    # Rename all treatment features
    dataset = dataset.rename(columns={'from_access': 'treatment'})
    # Rename all target features
    dataset = dataset.rename(columns={'to_access': 'response'})
    return dataset

def preparation_oot(dataset_oot: pd.DataFrame, label: str):
    # Remove Column
    dataset = dataset_oot.drop(['index', 'movement', 'fea_3', 'fea_11', 'fea_49', 'fea_118','to_trx_band','to_tenure'], axis=1)
    # Rename all treatment features
    dataset = dataset.rename(columns={'from_access': 'treatment'})
    # Rename all target features
    dataset = dataset.rename(columns={'to_access': 'response'})
    return dataset

def declare_target_class(dataset:pd.DataFrame):
    """Function for declare the target class"""   
    #CN:
    dataset.loc[(dataset.treatment == 'grape') & (dataset.response == 'gojek'),'target_class'] = 0 
    #CR:
    dataset.loc[(dataset.treatment == 'grape') & (dataset.response == 'grape'),'target_class'] = 1 
    #TN:
    dataset.loc[(dataset.treatment == 'gojek') & (dataset.response == 'grape'),'target_class'] = 2 
    #TR:
    dataset.loc[(dataset.treatment == 'gojek') & (dataset.response == 'gojek'),'target_class'] = 3 
    return dataset

def declare_target_class_oot(dataset:pd.DataFrame):
    """Function for declare the target class"""   
    #CN:
    dataset.loc[(dataset.treatment == 'grape') & (dataset.response == 'gojek'),'target_class'] = 0 
    #CR:
    dataset.loc[(dataset.treatment == 'grape') & (dataset.response == 'grape'),'target_class'] = 1 
    #TN:
    dataset.loc[(dataset.treatment == 'gojek') & (dataset.response == 'grape'),'target_class'] = 2 
    #TR:
    dataset.loc[(dataset.treatment == 'gojek') & (dataset.response == 'gojek'),'target_class'] = 3 
    return dataset

def encoding_target_class(dataset:pd.DataFrame):
    """Function for declare the target class"""
    #CN
    dataset['target_CN'] = 0 
    dataset.loc[(dataset.target_class == 0),'target_CN'] = 1
    #CR:
    dataset['target_CR'] = 0 
    dataset.loc[(dataset.target_class == 1),'target_CR'] = 1 
    #TN:
    dataset['target_TN'] = 0 
    dataset.loc[(dataset.target_class == 2),'target_TN'] = 1 
    #TR:
    dataset['target_TR'] = 0 
    dataset.loc[(dataset.target_class == 3),'target_TR'] = 1
    #remove target
    dataset = dataset.drop('target_class', axis=1)
    return dataset

def encoding_target_class_oot(dataset:pd.DataFrame):
    """Function for declare the target class"""
    #CN
    dataset['target_CN'] = 0 
    dataset.loc[(dataset.target_class == 0),'target_CN'] = 1
    #CR:
    dataset['target_CR'] = 0 
    dataset.loc[(dataset.target_class == 1),'target_CR'] = 1 
    #TN:
    dataset['target_TN'] = 0 
    dataset.loc[(dataset.target_class == 2),'target_TN'] = 1 
    #TR:
    dataset['target_TR'] = 0 
    dataset.loc[(dataset.target_class == 3),'target_TR'] = 1
    #remove target
    dataset = dataset.drop('target_class', axis=1)
    return dataset

def split_data(df_model:pd.DataFrame):
    """Split data into training data and testing data"""
    training, testing  = train_test_split(df_model, test_size=0.3, random_state=123)
    return training, testing

def feature_selection_CN(dataset:pd.DataFrame) -> pd.DataFrame:
    selected_features = mrmr_classif(dataset.drop(['response','treatment', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1), dataset.target_CN, K = 10)
    df_selected_features = dataset.loc[:, selected_features]
    df_model = pd.concat([df_selected_features, dataset.target_CN, dataset.response], axis=1)
    return df_model

def feature_selection_CR(dataset:pd.DataFrame) -> pd.DataFrame:
    selected_features = mrmr_classif(dataset.drop(['response','treatment', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1), dataset.target_CR, K = 10)
    df_selected_features = dataset.loc[:, selected_features]
    df_model = pd.concat([df_selected_features, dataset.target_CR, dataset.response], axis=1)
    return df_model

def feature_selection_TN(dataset:pd.DataFrame) -> pd.DataFrame:
    selected_features = mrmr_classif(dataset.drop(['response','treatment', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1), dataset.target_TN, K = 10)
    df_selected_features = dataset.loc[:, selected_features]
    df_model = pd.concat([df_selected_features, dataset.target_TN, dataset.response], axis=1)
    return df_model

def feature_selection_TR(dataset:pd.DataFrame) -> pd.DataFrame:
    selected_features = mrmr_classif(dataset.drop(['response','treatment', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1), dataset.target_TR, K = 10)
    df_selected_features = dataset.loc[:, selected_features]
    df_model = pd.concat([df_selected_features, dataset.target_TR, dataset.response], axis=1)
    return df_model

def upsample_CN(df_model_CN:pd.DataFrame):
    #FEATURE SELECTION
    one_ori_CN = df_model_CN.target_CN == 1
    zero_ori_CN = df_model_CN.target_CN == 0

    loop = 4500
    coe_atl_s3 = one_ori_CN.copy()
    df_ones_cn = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+22)
        df_ones_cn = sample.append(df_ones_cn) 

    loop = 5500
    coe_atl_s3 = zero_ori_CN.copy()
    df_zeros_cn = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+11)
        df_zeros_cn = sample.append(df_zeros_cn) 

    training_CN_boostrap = df_zeros_cn.append(df_ones_cn)
    return training_CN_boostrap

def upsample_CR(df_model_CR:pd.DataFrame):
    #FEATURE SELECTION
    one_ori_CR = df_model_CR.target_CR == 1
    zero_ori_CR = df_model_CR.target_CR == 0

    loop = 4500
    coe_atl_s3 = one_ori_CR.copy()
    df_ones_cr = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+22)
        df_ones_cr = sample.append(df_ones_cr) 

    loop = 5500
    coe_atl_s3 = zero_ori_CR.copy()
    df_zeros_cr = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+11)
        df_zeros_cr = sample.append(df_zeros_cr) 

    training_CR_boostrap = df_zeros_cr.append(df_ones_cr)
    return training_CR_boostrap

def upsample_TN(df_model_TN:pd.DataFrame):
    #FEATURE SELECTION
    one_ori_TN = df_model_TN.target_TN == 1
    zero_ori_TN = df_model_TN.target_TN == 0

    loop = 4500
    coe_atl_s3 = one_ori_TN.copy()
    df_ones_tn = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+22)
        df_ones_tn = sample.append(df_ones_tn) 

    loop = 5500
    coe_atl_s3 = zero_ori_TN.copy()
    df_zeros_tn = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+11)
        df_zeros_tn = sample.append(df_zeros_tn) 

    training_TN_boostrap = df_zeros_tn.append(df_ones_tn)
    return training_TN_boostrap

def upsample_TR(df_model_TR:pd.DataFrame):
    #FEATURE SELECTION
    one_ori_TR = df_model_TR.target_TR == 1
    zero_ori_TR = df_model_TR.target_TR == 0

    loop = 4500
    coe_atl_s3 = one_ori_TR.copy()
    df_ones_tr = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+22)
        df_ones_tr = sample.append(df_ones_tr) 

    loop = 5500
    coe_atl_s3 = zero_ori_TR.copy()
    df_zeros_tr = []
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 3, random_state = i+11)
        df_zeros_tr = sample.append(df_zeros_tr) 

    training_TR_boostrap = df_zeros_tr.append(df_ones_tr)
    return training_TR_boostrap

def split_xy_CN(training_CN_boostrap:pd.DataFrame, testing:pd.DataFrame):
    CN_X_train = training_CN_boostrap.drop(['target_CN'], axis = 1)
    CN_y_train = training_CN_boostrap.target_CN
    CN_X_test = testing.drop(['target_CN'], axis = 1)
    CN_y_test = testing.target_CN
    return CN_X_train, CN_y_train, CN_X_test, CN_y_test

def split_xy_CR(training:pd.DataFrame, testing:pd.DataFrame):
    CR_X_train = training.drop(['target_CR'], axis = 1)
    CR_y_train = training.target_CR
    CR_X_test = testing.drop(['target_CR'], axis = 1)
    CR_y_test = testing.target_CR
    return CR_X_train, CR_y_train, CR_X_test, CR_y_test

def split_xy_TN(training:pd.DataFrame, testing:pd.DataFrame):
    TN_X_train = training.drop(['target_TN'], axis = 1)
    TN_y_train = training.target_TN
    TN_X_test = testing.drop(['target_TN'], axis = 1)
    TN_y_test = testing.target_TN
    return TN_X_train, TN_y_train, TN_X_test, TN_y_test

def split_xy_TR(training:pd.DataFrame, testing:pd.DataFrame):
    TR_X_train = training.drop(['target_TR'], axis = 1)
    TR_y_train = training.target_TR
    TR_X_test = testing.drop(['target_TR'], axis = 1)
    TR_y_test = testing.target_TR
    return TR_X_train, TR_y_train, TR_X_test, TR_y_test

def modeling_CN(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """Machine learning process consists of 
    data training, and data testing process (i.e. prediction) with Catboost Algorithm
    """
    # prepare a new DataFrame
    training = pd.DataFrame(X_train).copy()
    testing = pd.DataFrame(X_test).copy()

    clf = CatBoostClassifier()
    params = {'iterations': [25, 30],
              'learning_rate': [0.001, 0.005, 0.01],
              'depth': [2, 3, 4],
              'loss_function': ['LogLoss', 'CrossEntropy'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
              'random_strength':[0,5,10],
              'random_seed': [33, 42]
             }
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True,
                                     multi_class='ovr')
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring = roc_auc_ovr_scorer, cv=5)
    clf_grid.fit(X_train, y_train)
    best_param = clf_grid.best_params_

    model_1 = CatBoostClassifier(iterations=best_param['iterations'],
                            learning_rate=best_param['learning_rate'],
                            depth=best_param['depth'],
                            loss_function=best_param['loss_function'],
                            l2_leaf_reg=best_param['l2_leaf_reg'],
                            eval_metric='Accuracy',
                            leaf_estimation_iterations=10,
                            use_best_model=True,
                            logging_level='Silent',
                            random_strength=best_param['random_strength'],
                            random_seed=best_param['random_seed']
                            )

    train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs,
    weight=sample_weight, group_id=group_id)
    
    model = model_1.fit(train_pool, eval_set = (X_test, y_test))

    prediction_train = model.predict(X_train)
    probability_train = model.predict_proba(X_train)
    training['prediction'] = prediction_train
    training['probability'] = probability_train[:,1] 

    prediction_test = model.predict(X_test)
    probability_test = model.predict_proba(X_test)
    testing['prediction'] = prediction_test
    testing['probability'] = probability_test[:,1]
          
    # add the churn and target class into dataframe as validation data
    training['label'] = y_train
    testing['label'] = y_test

    return training, testing, model

def modeling_CR(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """Machine learning process consists of 
    data training, and data testing process (i.e. prediction) with Catboost Algorithm
    """
    # prepare a new DataFrame
    training = pd.DataFrame(X_train).copy()
    testing = pd.DataFrame(X_test).copy()

    clf = CatBoostClassifier()
    params = {'iterations': [25, 30],
              'learning_rate': [0.001, 0.005, 0.01],
              'depth': [2, 3, 4],
              'loss_function': ['LogLoss', 'CrossEntropy'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
              'random_strength':[0,5,10],
              'random_seed': [33, 42]
             }
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True,
                                     multi_class='ovr')
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring = roc_auc_ovr_scorer, cv=5)
    clf_grid.fit(X_train, y_train)
    best_param = clf_grid.best_params_

    model_1 = CatBoostClassifier(iterations=best_param['iterations'],
                            learning_rate=best_param['learning_rate'],
                            depth=best_param['depth'],
                            loss_function=best_param['loss_function'],
                            l2_leaf_reg=best_param['l2_leaf_reg'],
                            eval_metric='Accuracy',
                            leaf_estimation_iterations=10,
                            use_best_model=True,
                            logging_level='Silent',
                            random_strength=best_param['random_strength'],
                            random_seed=best_param['random_seed']
                            )

    train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs,
    weight=sample_weight, group_id=group_id)
    
    model = model_1.fit(train_pool, eval_set = (X_test, y_test))

    prediction_train = model.predict(X_train)
    probability_train = model.predict_proba(X_train)
    training['prediction'] = prediction_train
    training['probability'] = probability_train[:,1] 

    prediction_test = model.predict(X_test)
    probability_test = model.predict_proba(X_test)
    testing['prediction'] = prediction_test
    testing['probability'] = probability_test[:,1]
          
    # add the churn and target class into dataframe as validation data
    training['label'] = y_train
    testing['label'] = y_test

    return training, testing, model

def modeling_TN(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """Machine learning process consists of 
    data training, and data testing process (i.e. prediction) with Catboost Algorithm
    """
    # prepare a new DataFrame
    training = pd.DataFrame(X_train).copy()
    testing = pd.DataFrame(X_test).copy()

    clf = CatBoostClassifier()
    params = {'iterations': [25, 30],
              'learning_rate': [0.001, 0.005, 0.01],
              'depth': [2, 3, 4],
              'loss_function': ['LogLoss', 'CrossEntropy'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
              'random_strength':[0,5,10],
              'random_seed': [33, 42]
             }
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True,
                                     multi_class='ovr')
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring = roc_auc_ovr_scorer, cv=5)
    clf_grid.fit(X_train, y_train)
    best_param = clf_grid.best_params_

    model_1 = CatBoostClassifier(iterations=best_param['iterations'],
                            learning_rate=best_param['learning_rate'],
                            depth=best_param['depth'],
                            loss_function=best_param['loss_function'],
                            l2_leaf_reg=best_param['l2_leaf_reg'],
                            eval_metric='Accuracy',
                            leaf_estimation_iterations=10,
                            use_best_model=True,
                            logging_level='Silent',
                            random_strength=best_param['random_strength'],
                            random_seed=best_param['random_seed']
                            )

    train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs,
    weight=sample_weight, group_id=group_id)
    
    model = model_1.fit(train_pool, eval_set = (X_test, y_test))

    prediction_train = model.predict(X_train)
    probability_train = model.predict_proba(X_train)
    training['prediction'] = prediction_train
    training['probability'] = probability_train[:,1] 

    prediction_test = model.predict(X_test)
    probability_test = model.predict_proba(X_test)
    testing['prediction'] = prediction_test
    testing['probability'] = probability_test[:,1]
          
    # add the churn and target class into dataframe as validation data
    training['label'] = y_train
    testing['label'] = y_test

    return training, testing, model

def modeling_TR(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """Machine learning process consists of 
    data training, and data testing process (i.e. prediction) with Catboost Algorithm
    """
    # prepare a new DataFrame
    training = pd.DataFrame(X_train).copy()
    testing = pd.DataFrame(X_test).copy()

    clf = CatBoostClassifier()
    params = {'iterations': [25, 30],
              'learning_rate': [0.001, 0.005, 0.01],
              'depth': [2, 3, 4],
              'loss_function': ['LogLoss', 'CrossEntropy'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
              'random_strength':[0,5,10],
              'random_seed': [33, 42]
             }
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True,
                                     multi_class='ovr')
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring = roc_auc_ovr_scorer, cv=5)
    clf_grid.fit(X_train, y_train)
    best_param = clf_grid.best_params_

    model_1 = CatBoostClassifier(iterations=best_param['iterations'],
                            learning_rate=best_param['learning_rate'],
                            depth=best_param['depth'],
                            loss_function=best_param['loss_function'],
                            l2_leaf_reg=best_param['l2_leaf_reg'],
                            eval_metric='Accuracy',
                            leaf_estimation_iterations=10,
                            use_best_model=True,
                            logging_level='Silent',
                            random_strength=best_param['random_strength'],
                            random_seed=best_param['random_seed']
                            )

    train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs,
    weight=sample_weight, group_id=group_id)
    
    model = model_1.fit(train_pool, eval_set = (X_test, y_test))

    prediction_train = model.predict(X_train)
    probability_train = model.predict_proba(X_train)
    training['prediction'] = prediction_train
    training['probability'] = probability_train[:,1] 

    prediction_test = model.predict(X_test)
    probability_test = model.predict_proba(X_test)
    testing['prediction'] = prediction_test
    testing['probability'] = probability_test[:,1]
          
    # add the churn and target class into dataframe as validation data
    training['label'] = y_train
    testing['label'] = y_test

    return training, testing, model

def evaluating_CN(training:pd.DataFrame, testing:pd.DataFrame):
    roc_training = "ROC Training: " + str(roc_auc_score(training['label'], training['probability'])) + "\n"
    roc_testing = "ROC Testing: " + str(roc_auc_score(testing['label'], testing['probability']))
    metrics = roc_training + roc_testing
    return metrics

def evaluating_CR(training:pd.DataFrame, testing:pd.DataFrame):
    roc_training = "ROC Training: " + str(roc_auc_score(training['label'], training['probability'])) + "\n"
    roc_testing = "ROC Testing: " + str(roc_auc_score(testing['label'], testing['probability']))
    metrics = roc_training + roc_testing
    return metrics

def evaluating_TN(training:pd.DataFrame, testing:pd.DataFrame):
    roc_training = "ROC Training: " + str(roc_auc_score(training['label'], training['probability'])) + "\n"
    roc_testing = "ROC Testing: " + str(roc_auc_score(testing['label'], testing['probability']))
    metrics = roc_training + roc_testing
    return metrics

def evaluating_TR(training:pd.DataFrame, testing:pd.DataFrame):
    roc_training = "ROC Training: " + str(roc_auc_score(training['label'], training['probability'])) + "\n"
    roc_testing = "ROC Testing: " + str(roc_auc_score(testing['label'], testing['probability']))
    metrics = roc_training + roc_testing
    return metrics

def oot_CN(dataset_oot:pd.DataFrame, model_CN:pd.DataFrame):
    X_CN_oot = dataset_oot.drop(['treatment','response','target_CN','target_CR', 'target_TN', 'target_TR'], axis=1)
    CN_prediction = model_CN.predict(X_CN_oot)
    CN_probability = model_CN.predict_proba(X_CN_oot)
    return CN_prediction, CN_probability 

def oot_CR(dataset_oot:pd.DataFrame, model_CR:pd.DataFrame):
    X_CR_oot = dataset_oot.drop(['treatment','response','target_CN','target_CR', 'target_TN', 'target_TR'], axis=1)
    CR_prediction = model_CR.predict(X_CR_oot)
    CR_probability = model_CR.predict_proba(X_CR_oot)
    return CR_prediction, CR_probability 

def oot_TN(dataset_oot:pd.DataFrame, model_TN:pd.DataFrame):
    X_TN_oot = dataset_oot.drop(['treatment','response','target_CN','target_CR', 'target_TN', 'target_TR'], axis=1)
    TN_prediction = model_TN.predict(X_TN_oot)
    TN_probability = model_TN.predict_proba(X_TN_oot)
    return TN_prediction, TN_probability 

def oot_TR(dataset_oot:pd.DataFrame, model_TR:pd.DataFrame):
    X_TR_oot = dataset_oot.drop(['treatment','response','target_CN','target_CR', 'target_TN', 'target_TR'], axis=1)
    TR_prediction = model_TR.predict(X_TR_oot)
    TR_probability = model_TR.predict_proba(X_TR_oot)
    return TR_prediction, TR_probability 

def combine_oot(dataset_oot:pd.DataFrame, CN_prediction:pd.DataFrame, CR_prediction:pd.DataFrame, TN_prediction:pd.DataFrame, TR_prediction:pd.DataFrame, CN_probability:pd.DataFrame, CR_probability:pd.DataFrame, TN_probability:pd.DataFrame, TR_probability:pd.DataFrame):
    X_data = dataset_oot.drop(['response', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1)
    y_data = dataset_oot.response
    CN_data = dataset_oot.target_CN
    CR_data = dataset_oot.target_CR
    TN_data = dataset_oot.target_TN
    TR_data = dataset_oot.target_TR
    oot = pd.DataFrame(X_data).copy()
    oot['response'] = y_data
    oot['target_CN'] = CN_data
    oot['target_CR'] = CR_data
    oot['target_TN'] = TN_data
    oot['target_TR'] = TR_data
    oot['prediction_target_CN'] = CN_prediction
    oot['prediction_target_CR'] = CR_prediction
    oot['prediction_target_TN'] = TN_prediction
    oot['prediction_target_TR'] = TR_prediction
    oot['proba_CN'] = CN_probability[:,1] 
    oot['proba_CR'] = CR_probability[:,1] 
    oot['proba_TN'] = TN_probability[:,1] 
    oot['proba_TR'] = TR_probability[:,1]

    oot['score_etu'] = oot.eval('\
        proba_CN/(proba_CN+proba_CR) \
        + proba_TR/(proba_TN+proba_TR) \
        - proba_TN/(proba_TN+proba_TR) \
        - proba_CR/(proba_CN+proba_CR)')
    oot['Decile'] = pd.qcut(oot['score_etu'], 10, labels=[i for i in range (10, 0, -1)])
    return oot

def combine(training:pd.DataFrame, testing:pd.DataFrame, model_CN:pd.DataFrame, model_CR:pd.DataFrame, model_TN:pd.DataFrame, model_TR:pd.DataFrame):
    training = training.reset_index().drop('index', axis=1)
    testing = testing.reset_index().drop('index', axis=1)
    X_train = training.drop(['response', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1)
    y_train = training.response
    CN_train = training.target_CN
    CR_train = training.target_CR
    TN_train = training.target_TN
    TR_train = training.target_TR
    X_test = testing.drop(['response', 'target_CN', 'target_CR', 'target_TN', 'target_TR'],axis=1)
    y_test = testing.response
    CN_test = testing.target_CN
    CR_test = testing.target_CR
    TN_test = testing.target_TN
    TR_test = testing.target_TR
    training_4k = pd.DataFrame(X_train).copy()
    training_4k['response'] = y_train
    testing_1k = pd.DataFrame(X_test).copy()
    testing_1k['response'] = y_test
    #CN
    train_pool_CN = Pool(X_train.drop('treatment', axis=1), CN_train)
    model_CN_train = model_CN.fit(train_pool_CN, eval_set=(X_test.drop('treatment', axis=1), CN_test))

    probability_CN_train = model_CN_train.predict_proba(X_train.drop('treatment', axis=1))

    prediction_CN_test = model_CN_train.predict(X_test.drop('treatment', axis=1))
    probability_CN_test = model_CN_train.predict_proba(X_test.drop('treatment', axis=1))

    #CR
    train_pool_CR = Pool(X_train.drop('treatment', axis=1), CR_train)
    model_CR_train = model_CR.fit(train_pool_CR, eval_set=(X_test.drop('treatment', axis=1), CR_test))

    probability_CR_train = model_CR_train.predict_proba(X_train.drop('treatment', axis=1))

    prediction_CR_test = model_CR_train.predict(X_test.drop('treatment', axis=1))
    probability_CR_test = model_CR_train.predict_proba(X_test.drop('treatment', axis=1))

    #TN
    train_pool_TN = Pool(X_train.drop('treatment', axis=1), TN_train)
    model_TN_train = model_TN.fit(train_pool_TN, eval_set=(X_test.drop('treatment', axis=1), TN_test))

    probability_TN_train = model_TN_train.predict_proba(X_train.drop('treatment', axis=1))

    prediction_TN_test = model_TN_train.predict(X_test.drop('treatment', axis=1))
    probability_TN_test = model_TN_train.predict_proba(X_test.drop('treatment', axis=1))

    #TR
    train_pool_TR = Pool(X_train.drop('treatment', axis=1), TR_train)
    model_TR_train = model_TR.fit(train_pool_TR, eval_set=(X_test.drop('treatment', axis=1), TR_test))

    probability_TR_train = model_TR_train.predict_proba(X_train.drop('treatment', axis=1))

    prediction_TR_test = model_TR_train.predict(X_test.drop('treatment', axis=1))
    probability_TR_test = model_TR_train.predict_proba(X_test.drop('treatment', axis=1))

    training_4k['target_CN'] = CN_train
    training_4k['target_CR'] = CR_train
    training_4k['target_TN'] = TN_train
    training_4k['target_TR'] = TR_train
    training_4k['proba_CN'] = probability_CN_train[:,1] 
    training_4k['proba_CR'] = probability_CR_train[:,1] 
    training_4k['proba_TN'] = probability_TN_train[:,1] 
    training_4k['proba_TR'] = probability_TR_train[:,1] 

    testing_1k['target_CN'] = CN_test
    testing_1k['target_CR'] = CR_test
    testing_1k['target_TN'] = TN_test
    testing_1k['target_TR'] = TR_test
    testing_1k['prediction_target_CN'] = prediction_CN_test
    testing_1k['prediction_target_CR'] = prediction_CR_test
    testing_1k['prediction_target_TN'] = prediction_TN_test
    testing_1k['prediction_target_TR'] = prediction_TR_test
    testing_1k['proba_CN'] = probability_CN_test[:,1] 
    testing_1k['proba_CR'] = probability_CR_test[:,1] 
    testing_1k['proba_TN'] = probability_TN_test[:,1] 
    testing_1k['proba_TR'] = probability_TR_test[:,1]

    training_4k['score_etu'] = training_4k.eval('\
        proba_CN/(proba_CN+proba_CR) \
        + proba_TR/(proba_TN+proba_TR) \
        - proba_TN/(proba_TN+proba_TR) \
        - proba_CR/(proba_CN+proba_CR)')

    testing_1k['score_etu'] = testing_1k.eval('\
        proba_CN/(proba_CN+proba_CR) \
        + proba_TR/(proba_TN+proba_TR) \
        - proba_TN/(proba_TN+proba_TR) \
        - proba_CR/(proba_CN+proba_CR)') 
    training_4k['Decile'] = pd.qcut(training_4k['score_etu'], 10, labels=[i for i in range (10, 0, -1)])
    testing_1k['Decile'] = pd.qcut(testing_1k['score_etu'], 10, labels=[i for i in range (10, 0, -1)])
    return training_4k, testing_1k
