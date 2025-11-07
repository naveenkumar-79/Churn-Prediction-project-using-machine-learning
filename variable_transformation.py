import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from log import setup_logging
logger = setup_logging('variable_transformation')
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings('ignore')
def quantile_trans(train_num,test_num):
    try:
        dummy_train_num = pd.DataFrame()
        dummy_test_num = pd.DataFrame()
        cols = ['TotalCharges', 'MonthlyCharges']
        dummy_train_num[cols] = train_num[cols].copy()
        dummy_test_num[cols] = test_num[cols].copy()
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        train_num[cols] = qt.fit_transform(train_num[cols])
        test_num[cols] = qt.transform(test_num[cols])
        for col in cols:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(train_num[col].dropna(), color='blue', fill=True)
            plt.title(f'KDE Plot for x_train{col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            plt.show()

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=train_num[col], color='skyblue')
            plt.title(f'Box Plot for x_train{col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            plt.show()

            plt.figure(figsize=(6, 6))
            stats.probplot(train_num[col], dist='norm', plot=plt)
            plt.title(f'Probability Plot  for x_train  {col}')
            plt.show()

        for col in cols:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(test_num[col].dropna(), color='blue', fill=True)
            plt.title(f'KDE Plot for x_test{col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            plt.show()

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=test_num[col], color='skyblue')
            plt.title(f'Box Plot for x_test{col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            plt.show()

            plt.figure(figsize=(6, 6))
            stats.probplot(test_num[col], dist='norm', plot=plt)
            plt.title(f'Probability Plot  for x_test  {col}')
            plt.show()
        return train_num, test_num
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
    # def box_cox(self):
    #     bct = BoxCoxTransformer(variables=['MonthlyCharges'])
    #     bct.fit(self.x_train_num[['MonthlyCharges']])
    #     self.x_train_num[['MonthlyCharges']] = bct.transform(self.x_train_num[['MonthlyCharges']])
    #     plt.figure(figsize=(10, 6))
    #     sns.kdeplot(self.x_train_num['MonthlyCharges'].dropna(), label='MonthlyCharges')
    #     plt.legend()
    #     plt.title("KDE Plot for Numeric Columns")
    #     plt.show()
    #     plt.figure(figsize=(6, 4))
    #     sns.boxplot(x=self.x_train_num['MonthlyCharges'], color='skyblue')
    #     plt.title(f'Box Plot of MonthlyCharges')
    #     plt.xlabel('MonthlyCharges')
    #     plt.grid(True, alpha=0.5)
    #     plt.show()
    #     plt.figure(figsize=(6, 6))
    #     stats.probplot(self.x_train_num['MonthlyCharges'], dist='norm', plot=plt)
    #     plt.title("Probability Plot for MonthlyCharges")
    #     plt.show()

    # def power(self):
    #
    #     from sklearn.preprocessing import PowerTransformer
    #     pt = PowerTransformer(method='yeo-johnson', standardize=True)
    #     pt.fit(self.x_train_num[['TotalCharges','MonthlyCharges']])
    #     cols=['TotalCharges','MonthlyCharges']
    #     for col in cols:
    #         plt.figure(figsize=(8, 4))
    #         sns.kdeplot(self.x_train_num[col].dropna(), color='blue', fill=True)
    #         plt.title(f'KDE Plot for x_train{col} using power')
    #         plt.xlabel(col)
    #         plt.grid(True, alpha=0.3)
    #         plt.show()
    #
    #         plt.figure(figsize=(6, 4))
    #         sns.boxplot(x=self.x_train_num[col], color='skyblue')
    #         plt.title(f'Box Plot for x_train{col} using power')
    #         plt.xlabel(col)
    #         plt.grid(True, alpha=0.3)
    #         plt.show()
    #
    #         plt.figure(figsize=(6, 6))
    #         stats.probplot(self.x_train_num[col], dist='norm', plot=plt)
    #         plt.title(f'Probability Plot  for x_train  {col} using power')
    #         plt.show()