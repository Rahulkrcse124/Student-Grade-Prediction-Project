import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('student.csv')


print('Total number of students: ', len(df))
print("Parameters are: ", df.columns)
df.info()


df = df.dropna()

label_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in label_columns:
    df[col] = le.fit_transform(df[col])

print("After encoding:")
print(df.dtypes)

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


sns.countplot(x="age", data=df)
plt.show()


most_correlated = df.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]  
print("Most correlated features: \n", most_correlated)


df = df.loc[:, most_correlated.index]
df.head()


X_train, X_test, y_train, y_test = train_test_split(df.drop('G3', axis=1), df['G3'], test_size=0.3, random_state=0)


lr = LinearRegression()
model = lr.fit(X_train, y_train)

print("Model Score (R^2):", lr.score(X_test, y_test)) 


predictions = lr.predict(X_test)


plt.scatter(y_test, predictions)
m, b = np.polyfit(y_test, predictions, 1)
plt.plot(y_test, m * y_test + b, color='red')
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.show()


user_input = {}  
for col in X_train.columns:  
    user_input[col] = float(input(f"Enter value for {col}: "))  
user_input_df = pd.DataFrame([user_input])


predicted_grade = lr.predict(user_input_df)
print("Predicted Grade:", predicted_grade[0])


df.head()
