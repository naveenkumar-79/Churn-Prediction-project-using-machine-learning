import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import logging
import sys
from numpy.random import logistic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from log import setup_logging
from handle_outlier import win
from model_building import common_algos, lr_algo
from variable_transformation import quantile_trans
from handling_missing_values import iterative_imputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from grid_search_parameters import checking_parameters
from sklearn.preprocessing import OrdinalEncoder
from feature_selection import ordinal_encoding,chi_square_test
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
logger=setup_logging('main')
import warnings
warnings.filterwarnings('ignore')

class cust_churn_pred:
    def __init__(self):
        try:

            self.df=pd.read_csv('C:\\Users\\MURALI\\OneDrive\\Desktop\\Internship Projects\\Chunk prediction\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
            self.new_df=pd.read_csv('C:\\Users\\MURALI\\OneDrive\\Desktop\\Internship Projects\\Chunk prediction\\Telco_Data_With_Tax_Gateway_Updated.csv')
            logger.info(self.df.isnull().sum())
            churn_index = self.df.columns.get_loc('Churn')
            self.df.insert(churn_index, 'sim', self.new_df['sim'])
            self.df.to_csv('updated_csv',index=False)
            logger.info(f'data shape {self.df.shape}')
            self.df=self.df.drop('customerID',axis=1)
            logger.info(self.df)
            self.df['TotalCharges']=self.df['TotalCharges'].apply(pd.to_numeric,errors='coerce')
            logger.info(f'datatypes of each column{self.df.dtypes}')
            self.num=self.df.select_dtypes(include=['number']).columns
            self.cat = self.df.select_dtypes(include=['object']).columns
            self.x=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def tenure_conv(self):
        import pandas as pd
        from dateutil.relativedelta import relativedelta
        base_date = pd.to_datetime("2015-01-01")
        self.df['JoinDate'] = [base_date + relativedelta(months=int(x) - 1) for x in self.df['tenure']]
        self.df['JoinYear'] = self.df['JoinDate'].dt.year
        self.df['JoinMonth'] = self.df['JoinDate'].dt.month_name()
        print(self.df[['tenure', 'JoinDate', 'JoinYear', 'JoinMonth']].head(10))
        print(self.df[['tenure', 'JoinDate', 'JoinYear', 'JoinMonth']].sample(5))
        self.df.to_csv('check2.csv',index = False)
        logger.info(f'dataset{self.df}')

    def iterative_imputation(self):
        try:
            self.x_train, self.x_test=iterative_imputer(self.x_train, self.x_test)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def cat_num_seperation(self):
        try:
            self.x_train = self.x_train.copy(deep=True)
            self.x_test = self.x_test.copy(deep=True)
            for col in ['SeniorCitizen', 'tenure']:
                if col in self.x_train.columns:
                    self.x_train[col] = self.x_train[col].astype('object')
                if col in self.x_test.columns:
                    self.x_test[col] = self.x_test[col].astype('object')
            self.x_train_num = self.x_train.select_dtypes(include=['number']).copy()
            self.x_train_cat = self.x_train.select_dtypes(include=['object', 'category']).copy()
            self.x_test_num = self.x_test.select_dtypes(include=['number']).copy()
            self.x_test_cat = self.x_test.select_dtypes(include=['object', 'category']).copy()
            logger.info(f'Numerical columns: {(self.x_train_num.columns)}')
            logger.info(f'Categorical columns: {(self.x_train_cat.columns)}')
            logger.info(f'missing values after data seperation{self.x_train_num.isnull().sum()}')
            logger.info(f'missing values after data seperation{self.x_test_num.isnull().sum()}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f"Issue in cat_num_seperation: {er_lin.tb_lineno} : due to : {er_msg}")

    def variable_trans(self):
        try:
            self.x_train_num, self.x_test_num = quantile_trans(self.x_train_num, self.x_test_num)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f"Issue in cat_num_seperation: {er_lin.tb_lineno} : due to : {er_msg}")

    def outlier_handling(self):
        try:
            self.x_train_num, self.x_test_num = win(self.x_train_num, self.x_test_num)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f"Issue in cat_num_seperation: {er_lin.tb_lineno} : due to : {er_msg}")


    def encoding(self):
        try:
            self.x_train_cat,self.x_test_cat,self.y_train,self.y_test=ordinal_encoding(self.x_train_cat,self.x_test_cat,self.y_train,self.y_test)
            self.x_train_cat,self.x_test_cat,self.y_train=chi_square_test(self.x_train_cat,self.x_test_cat,self.y_train)
            logger.info(f'{self.x_train_num.shape}')
            logger.info(f'{self.x_test_num.shape}')
            logger.info(f'{self.x_train_cat.shape}')
            logger.info(f'{self.x_test_cat.shape}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in chi_square_test: line {er_lin.tb_lineno} due to: {er_msg}")

    def merge_data(self):
        try:
            self.x_train_num.reset_index(drop=True, inplace=True)
            self.x_train_cat.reset_index(drop=True, inplace=True)
            self.x_test_num.reset_index(drop=True, inplace=True)
            self.x_test_cat.reset_index(drop=True, inplace=True)
            self.training_data = pd.concat([self.x_train_num, self.x_train_cat], axis=1)
            self.testing_data = pd.concat([self.x_test_num, self.x_test_cat], axis=1)
            logger.info(f'Training_data shape : {self.training_data.shape}')
            logger.info(f'Testing_data shape : {self.testing_data.shape}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def balance_data(self):
        try:
            logger.info(f'Total row for yes category in training data {self.training_data.shape[0]} was :{sum(self.y_train == 1)}')
            logger.info(f'Total row for No category in training data {self.training_data.shape[0]} was:{sum(self.y_train == 0)}')
            sm=SMOTE(random_state=42)
            self.training_data_res, self.y_train_res = sm.fit_resample(self.training_data,self.y_train)
            logger.info( f'Total row for Yes category in training data {self.training_data_res.shape[0]} was :{sum(self.y_train_res == 1)}')
            logger.info(f'Total row for No category in training data {self.training_data_res.shape[0]} was:{sum(self.y_train_res==0)}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')



    def feature_scaling(self):
        try:
            logger.info(f'{self.training_data_res.head(4)}')
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

            sc = StandardScaler()
            sc.fit(self.training_data_res[num_cols].to_numpy(dtype=float))
            self.training_data_res[num_cols] = pd.DataFrame(
                sc.transform(self.training_data_res[num_cols].to_numpy(dtype=float)),
                columns=num_cols,
                index=self.training_data_res.index
            )
            self.testing_data[num_cols] = pd.DataFrame(
                sc.transform(self.testing_data[num_cols].to_numpy(dtype=float)),
                columns=num_cols,
                index=self.testing_data.index
            )

            other_train = self.training_data_res.drop(num_cols, axis=1, errors='ignore')
            other_test = self.testing_data.drop(num_cols, axis=1, errors='ignore')

            self.training_data_res_b = pd.concat([other_train, self.training_data_res[num_cols]], axis=1)
            self.testing_data_res_b = pd.concat([other_test, self.testing_data[num_cols]], axis=1)

            # Save scaler
            with open('standard_scalar.pkl', 'wb') as t:
                pickle.dump(sc, t)

            logger.info(f'training data for the model: {self.training_data_res_b}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.exception(f'issue is:{er_lin.tb_lineno} due to: {er_msg}')

    def model_training(self):
        try:
            common_algos(self.training_data_res_b,self.y_train_res,self.testing_data_res_b,self.y_test)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'issue is:{er_lin.tb_lineno} due to: {er_msg}')

    # def parameters(self):
    #     self.training_data_res_b,self.y_train_res=checking_parameters(self.training_data_res_b,self.y_train_res)
    def best_model(self):
        try:
            self.lr_reg = LogisticRegression(C=100.0, class_weight= None, l1_ratio= None, max_iter= 100, multi_class= 'auto', n_jobs= None, penalty= 'l1', solver='liblinear')
            self.lr_reg.fit(self.training_data_res_b,self.y_train_res)
            logger.info(f'Test Accuracy LR : {accuracy_score(self.y_test, self.lr_reg.predict(self.testing_data_res_b))}')
            logger.info(f'confusion matrix : {confusion_matrix(self.y_test, self.lr_reg.predict(self.testing_data_res_b))}')
            logger.info(f'classification_report : {classification_report(self.y_test, self.lr_reg.predict(self.testing_data_res_b))}')
            with open('Churn Prediction.pkl','wb') as f:
                pickle.dump(self.lr_reg,f)
            with open('model_features.pkl', 'wb') as f:
                pickle.dump(self.training_data_res_b.columns.tolist(), f)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


if __name__=="__main__":
    obj=cust_churn_pred()
    obj.tenure_conv()
    obj.iterative_imputation()
    obj.cat_num_seperation()
    obj.variable_trans()
    obj.outlier_handling()
    obj.encoding()
    obj.merge_data()
    obj.balance_data()
    obj.feature_scaling()
    obj.model_training()
    #obj.parameters()
    obj.best_model()