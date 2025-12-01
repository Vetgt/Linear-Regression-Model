import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load dataset to dataframe pandas
file_path = "datasett.csv"
data = pd.read_csv(file_path)

# Detecting and Removing Outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (data < lower_bound) | (data > upper_bound)

# Count and display the number of outliers in each column
print("Number of Outliers in Each Column:")
print(outliers.sum())

# Remove outliers from the dataset
data_cleaned = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# Re-assign cleaned data to X and y
X = data_cleaned[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', '0V', 'DEC']]
y = data_cleaned['ANNUAL RAINFALL']

# Data Split: Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing independent variable
sc = StandardScaler()

# Standardizing X_train with fit_transform
X_train = sc.fit_transform(X_train)

# Standardizing X_test with transform
X_test = sc.transform(X_test)

# Model with regression Linear method
model = LinearRegression()

# Modeling variable X_train and y_train using fit()
model.fit(X_train, y_train)

# Making y_pred variable for predicting X_test
y_pred = model.predict(X_test)

# Evaluation Model with R2, MSE (Mean Squared Error), MAE, and MAPE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = (abs((y_test - y_pred) / y_test).mean()) * 100
accuracy = 100 - mape
mean_values = data_cleaned.mean()
median_values = data_cleaned.median()
std_deviation = data_cleaned.std()

# Generate a correlation matrix
correlation_matrix = data_cleaned.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Showing evaluation results
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))
print("R-squared (R2):", r2)
print("Accuracy (Approximation based on MAPE): {:.2f}%".format(accuracy))
print("Intercept (bias):", model.intercept_)
print("Coefficients:", model.coef_)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Mean Values for Each Column:")
print(mean_values)
print("\nMedian Values for Each Column:")
print(median_values)
print("\nStandard Deviation for Each Column:")
print(std_deviation)



# Scatter Plot: Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Annual Rainfall")
plt.ylabel("Predicted Annual Rainfall")
plt.title("Actual vs Predicted Annual Rainfall")
plt.grid()
plt.show()

# Visualizing relationship between 'YEAR' and 'ANNUAL RAINFALL'
plt.figure(figsize=(8, 6))
plt.scatter(data_cleaned['JAN'], data_cleaned['ANNUAL RAINFALL'], alpha=0.7, color='green')
plt.xlabel("JANUARY")
plt.ylabel("ANNUAL RAINFALL")
plt.title("JANUARY vs ANNUAL RAINFALL")
plt.grid()
plt.show()

# Visualizing relationship between 'YEAR' and 'ANNUAL RAINFALL'
plt.figure(figsize=(8, 6))
plt.scatter(data_cleaned['FEB'], data_cleaned['ANNUAL RAINFALL'], alpha=0.7, color='green')
plt.xlabel("FEBRUARY")
plt.ylabel("ANNUAL RAINFALL")
plt.title("FEBRUAY vs ANNUAL RAINFALL")
plt.grid()
plt.show()

# Visualizing relationship between 'YEAR' and 'ANNUAL RAINFALL'
plt.figure(figsize=(8, 6))
plt.scatter(data_cleaned['MAR'], data_cleaned['ANNUAL RAINFALL'], alpha=0.7, color='green')
plt.xlabel("MARCH")
plt.ylabel("ANNUAL RAINFALL")
plt.title("MARCH vs ANNUAL RAINFALL")
plt.grid()
plt.show()


# Scatter Plot: Actual vs Predicted values (detailed)
plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')
plt.scatter(range(len(y_test)), y_pred, color='#88c999', alpha=0.7, label='Predicted', edgecolors='none')
plt.title('Actual vs Predicted Values')
plt.xlabel('Data Point')
plt.ylabel('Target Variable')
plt.legend()
plt.show()

