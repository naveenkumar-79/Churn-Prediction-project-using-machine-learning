import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys




class churn:
    def churn_visual(self):
        try:
            self.df=pd.read_csv('C:\\Users\\MURALI\\OneDrive\\Desktop\\Internship Projects\\Chunk prediction\\WA_Fn-UseC_-Telco-Customer-Churn.csv')


            churn_counts = self.df['Churn'].value_counts()
            churn_labels = churn_counts.index
            churn_sizes = churn_counts.values
            plt.figure(figsize=(5, 3))
            plt.pie(churn_sizes, labels=churn_labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], explode=[0, 0.1],shadow=True)
            plt.title(' Churn Distribution')
            plt.show()

            churn_to_gender = self.df.groupby('gender')['Churn'].value_counts(normalize=True)*100
            print(churn_to_gender)

            senior_to_gender = self.df.groupby('gender')['SeniorCitizen'].value_counts(normalize=True) * 100
            print('gender w.r.to senior citizen',senior_to_gender)

            churn_to_dependents = self.df.groupby('Dependents')['Churn'].value_counts(normalize=True) * 100
            print(churn_to_dependents)

            Dependents = self.df[(self.df['Dependents']=='No')].groupby('gender')['SeniorCitizen'].value_counts(normalize=True) * 100
            print('Dependents comparision', Dependents)

            ak = sns.countplot(x='Dependents', hue='Churn', data=self.df)
            for i in ak.containers:
                ak.bar_label(i, label_type='center')
            plt.title("dependents w.r.to churn")
            plt.show()

            ao = sns.barplot(y='tenure', x='Churn', data=self.df)
            for i in ao.containers:
                ao.bar_label(i)
            plt.title("tenure")
            plt.show()

            churn_to_multipleLines = self.df.groupby('MultipleLines')['Churn'].value_counts(normalize=True) * 100
            print(churn_to_multipleLines)

            print('comparision between churn and monthly charges',self.df.groupby('Churn')['MonthlyCharges'].mean())

            print('senior citizens count',self.df['SeniorCitizen'].value_counts())

            self.seniors_churn=self.df['SeniorCitizen'].value_counts(normalize=True)*100
            print(self.seniors_churn)


            at=sns.countplot(x='Contract', hue='Churn', data=self.df)
            for i in at.containers:
                at.bar_label(i,label_type='center')
            plt.title("Churn by Contract Type")
            plt.show()

            ax=sns.barplot(x='gender', y='MonthlyCharges', hue='Churn', data=self.df)
            for i in ax.containers:
                ax.bar_label(i,label_type='center')
            plt.title("Average Monthly Charges by Gender and Churn")
            plt.show()

            ay=sns.barplot(y='MonthlyCharges',x='PaymentMethod', data=self.df)
            for i in ay.containers:
                ay.bar_label(i,label_type='center')
            plt.title("Average Monthly Charges by Payment Method")
            plt.show()


            billing_counts = self.df['PaperlessBilling'].value_counts()
            billing_labels = billing_counts.index
            billing_size = billing_counts.values
            plt.figure(figsize=(5, 3))
            plt.pie(billing_size, labels=billing_labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'],
                    explode=[0, 0.1],shadow=True)
            plt.title('paperlessbilling')
            plt.show()

            internet_counts = self.df['InternetService'].value_counts()
            internet_labels = internet_counts.index
            internet_size = internet_counts.values
            plt.figure(figsize=(5, 3))
            plt.pie(internet_size, labels=internet_labels, autopct='%1.1f%%', colors=['yellow', 'blue','green'], shadow=True)
            plt.title('Internet service provided')
            plt.show()

            InternetService = self.df[(self.df['InternetService'] == 'Fiber optic')].groupby('gender')['SeniorCitizen'].value_counts(
                normalize=True) * 100
            print('InternetService comparision', InternetService)






        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            print(er_ty, er_msg, er_lin)



if __name__=="__main__":
    try:
        obj=churn()
        obj.churn_visual()
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        print(er_ty, er_msg, er_lin)
