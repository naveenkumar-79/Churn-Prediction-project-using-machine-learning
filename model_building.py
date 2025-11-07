import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
from log import setup_logging
logger = setup_logging('model_building')
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def knn_algo(x_train,y_train,x_test,y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(x_train,y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test,knn_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test,knn_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test,knn_reg.predict(x_test))}')
        global knn_pred
        knn_pred = knn_reg.predict(x_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def nb_algo(x_train,y_train,x_test,y_test):
    try:
        nb_reg = GaussianNB()
        nb_reg.fit(x_train, y_train)
        logger.info(f'Test Accuracy NB : {accuracy_score(y_test, nb_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, nb_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test, nb_reg.predict(x_test))}')
        global nb_pred
        nb_pred = nb_reg.predict(x_test)
    except Exception as e:
        er_ty,er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
def lr_algo(x_train,y_train,x_test,y_test):
    try:
        lr_reg = LogisticRegression()
        lr_reg.fit(x_train, y_train)
        logger.info(f'Test Accuracy LR : {accuracy_score(y_test, lr_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, lr_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test, lr_reg.predict(x_test))}')
        global lr_pred
        lr_pred = lr_reg.predict(x_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def dt_algo(x_train,y_train,x_test,y_test):
    try:
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(x_train, y_train)
        logger.info(f'Test Accuracy DT : {accuracy_score(y_test, dt_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, dt_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test, dt_reg.predict(x_test))}')
        global dt_pred
        dt_pred = dt_reg.predict(x_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
def rf_algo(x_train,y_train,x_test,y_test):
    try:
        rf_reg = RandomForestClassifier(criterion='entropy',n_estimators=5)
        rf_reg.fit(x_train, y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, rf_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, rf_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test, rf_reg.predict(x_test))}')
        global rf_pred
        rf_pred = rf_reg.predict(x_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
def svm_algo(x_train,y_train,x_test,y_test):
    try:
        svm_reg=SVC(C=1)
        svm_reg.fit(x_train,y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, svm_reg.predict(x_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, svm_reg.predict(x_test))}')
        logger.info(f'classification_report : {classification_report(y_test, svm_reg.predict(x_test))}')
        global svm_pred
        svm_pred=svm_reg.predict(x_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def best_model_using_acu_roc(x_train,y_train, x_test, y_test):
    try:
        knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_pred)
        nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, nb_pred)
        lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_pred)
        dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_pred)
        rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_pred)
        svm_fpr,svm_tpr,svm_thre=roc_curve(y_test,svm_pred)

        knn_auc = auc(knn_fpr, knn_tpr)
        nb_auc = auc(nb_fpr, nb_tpr)
        lr_auc = auc(lr_fpr, lr_tpr)
        dt_auc = auc(dt_fpr, dt_tpr)
        rf_auc = auc(rf_fpr, rf_tpr)
        svm_auc=auc(svm_fpr,svm_tpr)

        logger.info(f'KNN:-{auc(knn_fpr,knn_tpr)}')
        logger.info(f'NB:-{auc(nb_fpr, nb_tpr)}')
        logger.info(f'lr:-{auc(lr_fpr, lr_tpr)}')
        logger.info(f'dt:-{auc(dt_fpr, dt_tpr)}')
        logger.info(f'rf:-{auc(rf_fpr, rf_tpr)}')
        logger.info(f'svm:-{auc(svm_fpr,svm_tpr)}')

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Auc ROC Curve - All Models")
        plt.plot([0, 1], [0, 1], "k--")

        plt.plot(knn_fpr, knn_tpr, color='r', label=f"KNN (AUC = {knn_auc:.3f})")
        plt.plot(nb_fpr, nb_tpr, color='blue', label=f"NB (AUC = {nb_auc:.3f})")
        plt.plot(lr_fpr, lr_tpr, color='green', label=f"LR (AUC = {lr_auc:.3f})")
        plt.plot(dt_fpr, dt_tpr, color='black', label=f"DT (AUC = {dt_auc:.3f})")
        plt.plot(rf_fpr, rf_tpr, color='orange', label=f"RF (AUC = {rf_auc:.3f})")
        plt.plot(svm_fpr, svm_tpr, color='pink', label=f"SVM (AUC = {svm_auc:.3f})")

        plt.legend(loc=0)
        plt.show()
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
def common_algos(x_train,y_train,x_test,y_test):
    try:
        logger.info('----knn--------')
        knn_algo(x_train, y_train, x_test, y_test)
        logger.info('----NB--------')
        nb_algo(x_train, y_train, x_test, y_test)
        logger.info('----LR--------')
        lr_algo(x_train, y_train, x_test, y_test)
        logger.info('----dt--------')
        dt_algo(x_train, y_train, x_test, y_test)
        logger.info('----rf--------')
        rf_algo(x_train, y_train, x_test, y_test)
        logger.info('----svm--------')
        svm_algo(x_train, y_train, x_test, y_test)
        logger.info(f'--------AUC--ROC----------')
        best_model_using_acu_roc(x_train, y_train, x_test, y_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

