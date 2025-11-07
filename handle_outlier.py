import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import logging
from log import setup_logging
logger = setup_logging('handle_outlier')
from feature_engine.outliers import Winsorizer
import warnings
warnings.filterwarnings('ignore')
def win(train_num,test_num):
    try:
        winsor = Winsorizer(
            capping_method='gaussian',
            tail='both',
            fold=2.5,
            variables=['MonthlyCharges', 'TotalCharges'])
        winsor.fit(train_num)
        winsor.fit(test_num)
        train_num = winsor.transform(train_num)
        test_num = winsor.transform(test_num)
        cols = ['MonthlyCharges', 'TotalCharges']
        for col in cols:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(train_num[col].dropna(), color='blue', fill=True)
            plt.title(f'KDE Plot for x_train using winsorlizer{col}')
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
        return train_num,test_num
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')