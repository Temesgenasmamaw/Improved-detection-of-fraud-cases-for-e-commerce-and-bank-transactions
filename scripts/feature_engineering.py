import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataFeatureEngineering:
        def feature_engineering(self,df):
            # Transaction frequency (number of transactions by user_id)
            df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
            
            # Time-based features
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.weekday
            
            return df
        
        def normalize_scale_features(self,df):
            scaler = StandardScaler()
            
            df[['purchase_value', 'transaction_count']] = scaler.fit_transform(df[['purchase_value', 'transaction_count']])
            
            return df
        
        def one_hot_encode(self,df):
            columns_to_encode = ['source', 'browser','sex']
            # Perform one-hot encoding on the specified columns
            df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
            
            # Get the newly created one-hot encoded columns (those that contain the original column names)
            encoded_columns = [col for col in df_encoded.columns if any(c in col for c in columns_to_encode)]
    
            # Convert the one-hot encoded columns to integer (0 and 1)
            df_encoded[encoded_columns] = df_encoded[encoded_columns].astype(int)
            
            return df_encoded
        
            
        def frequency_encode_country(self,df):
            # Calculate the frequency of each country
            freq_encoding = df['country'].value_counts() / len(df)
            
            # Map the country names to their frequency
            df['country' + '_encoded'] = df['country'].map(freq_encoding)
            
            # Optionally, drop the original 'country' column
            df = df.drop(columns=['country'])
            
            return df
