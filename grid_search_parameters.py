import pandas as pd
import sys
import logging
from log import setup_logging
logger = setup_logging('grid_search_parameters')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
def checking_parameters(training_data_res_b,y_train_res):
    try:
        sample_data = training_data_res_b.head(200)
        sample_dep = y_train_res[:200]
        logger.info(f'sample_data{sample_data}')
        lr_reg = LogisticRegression()
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 200, 500, 1000],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'n_jobs': [None, -1],
            'l1_ratio': [None, 0.0, 0.25, 0.5, 0.75, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=lr_reg,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5)
        grid_search.fit(training_data_res_b,y_train_res)
        logger.info(f'Best Parameters:{grid_search.best_params_}')
        logger.info(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')
        return training_data_res_b, y_train_res
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')