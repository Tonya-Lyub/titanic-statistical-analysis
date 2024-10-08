import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a local file
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows to check if the dataset is loaded correctly
print(df.head())

# Check for missing values in the dataset
print("Missing values in the data:")
print(df.isnull().sum())

# Fill missing values in the 'age' column with the median value
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing values in the 'embarked' column with the most frequent value (mode)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop the 'cabin' column as it has too many missing values
df.drop(['cabin'], axis=1, inplace=True)

# Plot the age distribution of passengers
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Comment: The histogram shows the distribution of passengers' ages, with the KDE curve providing an estimate of the density.

# Box plot to visualize fare distribution by passenger class
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# Comment: The box plot shows how fares vary across different passenger classes. We can identify outliers, especially for higher fare values.

# Bar chart of survival counts by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Comment: The bar chart visualizes the survival counts separated by gender. It highlights the higher survival rate for females compared to males.

# Filter numeric columns for correlation calculation
numeric_cols = df.select_dtypes(include=[np.number])

# Correlation heatmap of numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Comment: The heatmap shows the correlations between numerical features. Strong correlations can be seen between features such as 'fare' and 'pclass'.

# Bar plot of embarked location frequencies
plt.figure(figsize=(10, 6))
sns.countplot(x='embarked', data=df)
plt.title('Frequency of Embarked Locations')
plt.xlabel('Embarked Location')
plt.ylabel('Count')
plt.show()

# Comment: The bar plot shows the distribution of passengers across the embarkation points (Cherbourg, Queenstown, Southampton).
