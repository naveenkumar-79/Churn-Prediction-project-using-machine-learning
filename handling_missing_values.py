import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import logging
from log import setup_logging
logger = setup_logging('handling_missing_values')
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')
def iterative_imputer(train,test):
    try:
        initial_train = pd.DataFrame()
        initial_test = pd.DataFrame()
        initial_train['Total_charges'] = train['TotalCharges']
        initial_test['Total_charges'] =test['TotalCharges']
        decision_tree_estimator = DecisionTreeRegressor(random_state=42, max_depth=5)
        imputer = IterativeImputer(estimator=decision_tree_estimator, random_state=42)

        train['TotalCharges'] = imputer.fit_transform(train[['TotalCharges']])
        test['TotalCharges'] = imputer.fit_transform(test[['TotalCharges']])

        logger.info(f'checking x_train{train.isnull().sum()}')
        logger.info(f'checking x_test{test.isnull().sum()}')
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 2, 1)
        initial_train['Total_charges'].hist(bins=30, color='red', label=f"Before-{initial_train['Total_charges'].std()}")
        plt.title("Before Imputation x_train (Iterative Imputer) ")
        plt.legend()
        plt.subplot(2, 2, 2)
        train['TotalCharges'].hist(bins=30, color='blue', label=f"after-{train['TotalCharges'].std()}")
        plt.title("After Imputation x_train")
        plt.legend()
        plt.subplot(2, 2, 3)
        initial_test['Total_charges'].hist(bins=30, color='red',
                                                             label=f"Before-{initial_test['Total_charges'].std()}")
        plt.title("Before Imputation x_test")
        plt.legend()
        plt.subplot(2, 2, 4)
        test['TotalCharges'].hist(bins=30, color='blue', label=f"after-{test['TotalCharges'].std()}")
        plt.title("After Imputation x_test")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return train,test
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    # def RandomSampleImputer(self):
    #     try:
    #         self.initial_xtrain = pd.DataFrame()
    #         self.initial_xtest = pd.DataFrame()
    #         self.initial_xtrain['Total_charges'] = self.x_train['TotalCharges']
    #         self.initial_xtest['Total_charges'] = self.x_test['TotalCharges']
    #         self.random_simple = RandomSampleImputer()
    #         self.x_train = self.random_simple.fit_transform(self.x_train)
    #         self.x_test = self.random_simple.fit_transform(self.x_test)
    #         print(f'checking x_train{self.x_train.isnull().sum()}')
    #         print(f'checking x_test{self.x_test.isnull().sum()}')
    #
    #         logger.info(f'x_train data{self.x_train}')
    #         logger.info(f'x_test data{self.x_test}')
    #         #visualization
    #         plt.figure(figsize=(8,4))
    #         plt.subplot(2, 2, 1)
    #         self.initial_xtrain['Total_charges'].hist(bins=30, color='red',
    #                                                               label=f"Before-{self.initial_xtrain['Total_charges'].std()}")
    #         plt.title("Before Imputation x_train (RandomSampleImputer) ")
    #         plt.legend()
    #         plt.subplot(2, 2, 2)
    #         self.x_train['TotalCharges'].hist(bins=30, color='blue', label=f"after-{self.x_test['TotalCharges'].std()}")
    #         plt.title("After Imputation x_train")
    #         plt.legend()
    #         plt.subplot(2, 2, 3)
    #         self.initial_xtest['Total_charges'].hist(bins=30, color='red',
    #                                                              label=f"Before-{self.initial_xtest['Total_charges'].std()}")
    #         plt.title("Before Imputation x_test")
    #         plt.legend()
    #         plt.subplot(2, 2, 4)
    #         self.x_test['TotalCharges'].hist(bins=30, color='blue', label=f"after-{self.x_test['TotalCharges'].std()}")
    #         plt.title("After Imputation x_test")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #
    #     except Exception as e:
    #         er_ty, er_msg, er_lin = sys.exc_info()
    #         logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
    # def knn_imputation(self):
    #     try:
    #         self.initial_xtrain = pd.DataFrame()
    #         self.initial_xtest = pd.DataFrame()
    #         self.initial_xtrain['Total_charges'] = self.x_train['TotalCharges']
    #         self.initial_xtest['Total_charges'] = self.x_test['TotalCharges']
    #         imputer = KNNImputer(n_neighbors=3, weights='uniform')
    #         self.x_train['TotalCharges'] = imputer.fit_transform(self.x_train[['TotalCharges']])
    #         self.x_test['TotalCharges'] = imputer.fit_transform(self.x_test[['TotalCharges']])
    #         print(f'checking x_train{self.x_train.isnull().sum()}')
    #         print(f'checking x_test{self.x_test.isnull().sum()}')
    #
    #         logger.info(f'x_train data{self.x_train}')
    #         logger.info(f'x_test data{self.x_test}')
    #         # visualization
    #         plt.figure(figsize=(8, 4))
    #         plt.subplot(2, 2, 1)
    #         self.initial_xtrain['Total_charges'].hist(bins=30, color='red',
    #                                                   label=f"Before-{self.initial_xtrain['Total_charges'].std()}")
    #         plt.title("Before Imputation x_train (KNN) ")
    #         plt.legend()
    #         plt.subplot(2, 2, 2)
    #         self.x_train['TotalCharges'].hist(bins=30, color='blue', label=f"after-{self.x_test['TotalCharges'].std()}")
    #         plt.title("After Imputation x_train")
    #         plt.legend()
    #         plt.subplot(2, 2, 3)
    #         self.initial_xtest['Total_charges'].hist(bins=30, color='red',
    #                                                  label=f"Before-{self.initial_xtest['Total_charges'].std()}")
    #         plt.title("Before Imputation x_test")
    #         plt.legend()
    #         plt.subplot(2, 2, 4)
    #         self.x_test['TotalCharges'].hist(bins=30, color='blue', label=f"after-{self.x_test['TotalCharges'].std()}")
    #         plt.title("After Imputation x_test")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #     except Exception as e:
    #         er_ty, er_msg, er_lin = sys.exc_info()
    #         logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')