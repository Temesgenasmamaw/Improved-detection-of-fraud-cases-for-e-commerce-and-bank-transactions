import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


class DataVisualization:


    def __init__(self, fraud_df, numerical_features, categorical_features, target_col):

        self.fraud_df = fraud_df
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        # Calculate the time difference in minutes
        self.fraud_df['purchase_delay'] = (self.fraud_df['purchase_time'] - self.fraud_df['signup_time']).dt.total_seconds() / 60

    def _plot_numerical_histograms(self):
        """
        Plots histograms for the numerical features
        """
        plt.figure(figsize=(15, 4))
        for i, feature in enumerate(self.numerical_features, 1):
            plt.subplot(1, len(self.numerical_features), i)
            sns.histplot(self.fraud_df[feature], bins=30, kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # Plot Categorical_features
    def _plot_categorical(self):

        plt.figure(figsize=(15, 4))
        for i, feature in enumerate(self.categorical_features, 1):
            plt.subplot(1, len(self.categorical_features), i)
            order = self.fraud_df[feature].value_counts().index
            sns.countplot(data=self.fraud_df, x=feature, order=order, hue=feature, palette='viridis')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
        
    def _boxplot(self, x, y):

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=self.fraud_df, x=x, y=y)
        plt.title(f'{y.capitalize()} by {x.capitalize()}')
        plt.xlabel(x.capitalize())
        plt.ylabel(y.capitalize())
        plt.show()
        
    def plot_relationship(self):        

        plt.figure(figsize=(8, 4))
        sns.scatterplot(data=self.fraud_df, x='age', y='purchase_value', alpha=0.5)
        plt.title('Purchase Value vs Age')
        plt.xlabel('Age')
        plt.ylabel('Purchase Value')
        plt.show()
        print("Successfully plotted purchase_value vs age scatter plot.")

    
    def plot_source_vs_browser_heatmap(self):

            print("Plotting source vs browser heatmap.")
            source_browser_counts = pd.crosstab(self.fraud_df['source'], self.fraud_df['browser'])
            plt.figure(figsize=(10, 4))
            sns.heatmap(source_browser_counts, annot=True, fmt='d', cmap='Blues')
            plt.title('Source vs Browser')
            plt.xlabel('Browser')
            plt.ylabel('Source')
            plt.show()
            print("Successfully plotted source vs browser heatmap.")


    def plot_distribution_by_class(self):

        print("Plotting categorical variables distributions by class.")
        try:
            # Create a figure with two rows and three columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle('Distributions of Categorical Features by Class', fontsize=16)
            sns.set_style("whitegrid")

            # Define the features to plot and their titles
            features = ['sex', 'source', 'browser']
            titles = ['Sex Distribution by Class', 'Source Distribution by Class', 'Browser Distribution by Class']

            # Loop through the features and create count plots
            for ax, feature, title in zip(axes.flatten(), features, titles):
                sns.countplot(data=self.fraud_df, x=feature, hue=self.target_col, ax=ax, palette='viridis' if feature == 'sex' else 'muted')
                ax.set_title(title)
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel('Count')

            # Remove any unused axes
            for i in range(len(features), len(axes.flatten())):
                fig.delaxes(axes.flatten()[i])

            # Adjust layout for better spacing
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust space to accommodate the main title
            plt.show()

            print("Successfully plotted multiple distributions by class.")
        except Exception as e:
            print(f"Error in plotting multiple distributions by class: {e}")

            
    def plot_pairwise_relationships(self):
        """
        Plots pairwise relationships among numerical features grouped by the target variable.
        """
        sns.pairplot(self.fraud_df, vars=self.numerical_features, hue=self.target_col, palette='husl', plot_kws={'alpha': 0.6})
        plt.suptitle('Pair Plot of Numerical Features by Class', y=1.02)
        plt.show()

    def plot_browser_usage(self):
        """
        Plots a strip plot showing browser usage and purchase values grouped by class.
        """
        plt.figure(figsize=(12, 4))
        sns.stripplot(data=self.fraud_df, x='browser', y='purchase_value', hue='class', dodge=True, palette='viridis')
        plt.title('Browser Usage and Purchase Value by Class')
        plt.xlabel('Browser')
        plt.ylabel('Purchase Value')
        plt.legend(title='Class')
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Displays a heatmap of the correlation matrix including the target variable.
        """
        sns.set_style("white")
        correlation_matrix = self.fraud_df.corr()
        plt.figure(figsize=(8, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix Including Target Variable')
        plt.show()

    def plot_sunburst_chart(self):
        """
        Visualizes a sunburst chart for source, browser, and class hierarchy using Plotly.
        """
        fig = px.sunburst(self.fraud_df, path=['source', 'browser', self.target_col], color=self.target_col,
                          color_discrete_map={0: 'lightblue', 1: 'red'},
                          title="Source, Browser, and Class Hierarchy")
        fig.show()
    
    
    def plot_purchase_delay_distribution(self):
        """
        Plots the distribution of purchase delays (time between signup and purchase).
        """
        try:
            print("Plotting the distribution of purchase delays.")
            plt.figure(figsize=(10, 5))
            sns.histplot(self.fraud_df['purchase_delay'], bins=30, kde=True)
            plt.title('Distribution of Purchase Delays (Minutes)')
            plt.xlabel('Purchase Delay (Minutes)')
            plt.ylabel('Frequency')
            plt.show()
            print("Successfully plotted the distribution of purchase delays.")
        except Exception as e:
            print(f"Error in plotting purchase delay distribution: {e}")
    
    def plot_purchase_patterns_over_time(self):
        """
        Plots the purchase counts over different times (hour of the day and day of the week).
        """
        try:
            print("Plotting purchase patterns over time.")
            # Extract hour and day of the week
            self.fraud_df['hour_of_day'] = self.fraud_df['purchase_time'].dt.hour
            self.fraud_df['day_of_week'] = self.fraud_df['purchase_time'].dt.day_name()

            # Plot purchases by hour of the day
            plt.figure(figsize=(12, 4))
            sns.countplot(x='hour_of_day', data=self.fraud_df, hue='hour_of_day', legend=False, palette='coolwarm')
            plt.title('Purchases by Hour of the Day')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Purchases')
            plt.show()

            # Plot purchases by day of the week
            plt.figure(figsize=(12, 4))
            sns.countplot(x='day_of_week', data=self.fraud_df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], hue='day_of_week', palette='muted')
            plt.title('Purchases by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Number of Purchases')
            plt.show()
            
            print("Successfully plotted purchase patterns over time.")
        except Exception as e:
            print(f"Error in plotting purchase patterns over time: {e}")
    
    def plot_purchase_value_vs_delay(self):
        """
        Plots the relationship between purchase value and purchase delay.
        """
        try:
            print("Plotting the relationship between purchase value and purchase delay.")
            plt.figure(figsize=(10, 4))
            sns.scatterplot(data=self.fraud_df, x='purchase_delay', y='purchase_value', hue='class', palette='coolwarm', alpha=0.6)
            plt.title('Purchase Value vs. Purchase Delay')
            plt.xlabel('Purchase Delay (Minutes)')
            plt.ylabel('Purchase Value')
            plt.legend(title='Class (0: Non-Fraud, 1: Fraud)')
            plt.show()
            print("Successfully plotted purchase value vs. delay.")
        except Exception as e:
            print(f"Error in plotting purchase value vs delay: {e}")