import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDataPreprocessing:
    def __init__(self):
            self.fraud_data_path = '../data/Fraud_Data.csv'
            self.ip_country_path = '../data/IpAddress_to_Country.csv'
            self.creditcard_data_path='../data/creditcard.csv'
    def load_dataset(self):
            fraud_data = pd.read_csv(self.fraud_data_path)
            ip_country_data = pd.read_csv(self.ip_country_path)
            creditcard_data= pd.read_csv(self.creditcard_data_path)
            return fraud_data, ip_country_data, creditcard_data
    def data_overview(self,df):
            num_rows = df.shape[0]
            num_columns = df.shape[1]
            data_types = df.dtypes

            print(f"Number of rows:{num_rows}")
            print(f"Number of columns:{num_columns}")
            print(f"Data types of each column:\n{data_types}")
    def data_cleaning(self,df):
       df['signup_time'] = pd.to_datetime(df['signup_time'],errors='coerce')
       df['purchase_time'] = pd.to_datetime(df['purchase_time'],errors='coerce')
       
       print("Data type after conversion!\n")
       print(df.dtypes)
     
       # Check for duplicate rows in the DataFrame
       duplicate_count = df.duplicated().sum()
       print(f"Number of duplicate rows: {duplicate_count}")
       # Remove duplicates
       if duplicate_count:
            df.drop_duplicates(inplace=True)
