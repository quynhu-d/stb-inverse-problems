import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

def lsvc_classifier(X_train, y_train, X_test, y_test):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, y_train)

    train_score = linear_svc.score(X_train, y_train)
    test_score = linear_svc.score(X_test, y_test)
    
    b_pred_svc = linear_svc.decision_function(X_test)
    auc_roc_svc_ = roc_auc_score(y_test, b_pred_svc)
    auc_pr_svc_ = average_precision_score(y_test, b_pred_svc)
    
    return linear_svc, {
        'train_acc': train_score, 'test_acc': test_score,
        'auc_roc_test': auc_roc_svc_, 'auc_pr_test': auc_pr_svc_
    }
    
def svc_classifier(X_train, y_train, X_test, y_test):
    svc_pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('svc', LinearSVC(dual=False))
    ])
    reg_Cs = np.logspace(-5, 1, 20)
    linear_svc = GridSearchCV(svc_pipeline, {"svc__C": reg_Cs}, cv=10)    # chooses best by score estimate
    model = linear_svc.fit(X_train, y_train)

    best_model_svc = linear_svc.best_estimator_
    train_score = best_model_svc[1].score(X_train, y_train)
    test_score = best_model_svc[1].score(X_test, y_test)
    
    b_pred_svc = best_model_svc.decision_function(X_test)
    auc_roc_svc_ = roc_auc_score(y_test, b_pred_svc)
    auc_pr_svc_ = average_precision_score(y_test, b_pred_svc)
    
    return best_model_svc, {
        'train_acc': train_score, 'test_acc': test_score,
        'auc_roc_test': auc_roc_svc_, 'auc_pr_test': auc_pr_svc_
    }

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product


def decision_tree_classifier(
    X_train, y_train, X_test, y_test, 
    depth_grid=range(3, 16), samples_leaf_grid=range(1, 5), random_forest=False
):
    if random_forest:
        model = RandomForestClassifier()
    else:
        model = DecisionTreeClassifier()
    model_gscv = GridSearchCV(model, {"max_depth": depth_grid, "min_samples_leaf": samples_leaf_grid}, cv=10)    # chooses best by score estimate
    model = model_gscv.fit(X_train, y_train)

    best_model = model.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    return best_model, {'train_acc': train_score, 'test_acc': test_score}

def full_pipeline(df, method='wishart', features=None):
    df_train = df.query('part == \'train\'')
    df_test = df.query('part == \'test\'')
    
    if method in ['wishart', 'kmeans', 'fcmeans']:
        features = ['mean_icd', 'min_icd', 'max_icd']
    elif method in ['fuzzy', 'wishart_w_noise']:
        features = ['mean_icd', 'min_icd', 'max_icd', 'noise', 'n_clusters']
    elif method == 'ec':
        features = ['entropy', 'complexity']
    elif method == 'all':
        features = [
            'entropy', 'complexity',
            'ws_min_icd', 'ws_max_icd', 'ws_mean_icd',
            'fws_min_icd', 'fws_max_icd', 'fws_mean_icd', 'noise', 'n_clusters',
            'km_min_icd', 'km_max_icd', 'km_mean_icd',
            'fcm_min_icd', 'fcm_max_icd', 'fcm_mean_icd'
        ]
    elif method == 'tda':
        features = [
            'PE_0', 'PE_1', 'NoP_0', 'NoP_1', 'A_bottleneck_0', 'A_bottleneck_1',
            'A_wasserstein_0', 'A_wasserstein_1', 'A_landscape_0', 'A_landscape_1',
            'A_persistence_image_0', 'A_persistence_image_1'
        ]
    elif method == 'custom':
        assert features is not None
        
    X_train = df_train[features]
    y_train = df_train.text_type == 'lit'

    X_test = df_test[features]
    y_test = df_test.text_type == 'lit'
    
    res_svc = svc_classifier(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)
    res_lsvc = lsvc_classifier(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)
    res_dt = decision_tree_classifier(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)
    res_rf = decision_tree_classifier(
        X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
        samples_leaf_grid=range(1, 11),
        random_forest=True
    )
    
    res_clf = {}

    params_ = res_dt[0].get_params()
    params_dt = {'depth': params_['max_depth'], 'samples_leaf': params_['min_samples_leaf']}
    res_clf['decision_tree'] = {'params': params, 'model': res_dt[0]} | res_dt[1]   
    
    params_ = res_rf[0].get_params()
    params_rf = {'depth':params_['max_depth'], 'samples_leaf': params_['min_samples_leaf']}
    res_clf['random_forest'] = {'params': params, 'model': res_rf[0]} | res_rf[1]    

    params = {'C': res_svc[0][1].C}
    res_clf['svc'] = {'params': params, 'model': res_svc[0]} | res_svc[1]
    res_clf['lsvc'] = {'model': res_lsvc[0]} | res_lsvc[1]
    
    return res_clf