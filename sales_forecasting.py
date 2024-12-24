# src/sales_forecasting.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)


n_rows = 500

item_weight = np.random.uniform(5, 25, n_rows)

item_fat_content = np.random.choice(['Low Fat', 'Regular'], size=n_rows)

item_visibility = np.random.uniform(0, 1, n_rows)

item_mrp = np.random.uniform(50, 300, n_rows)


sales = 100 + (item_weight * 2) + (item_visibility * 150) + (item_mrp * 1.5) + np.random.normal(0, 50, n_rows)

df = pd.DataFrame({
    'Item_Weight': item_weight,
    'Item_Fat_Content': item_fat_content,
    'Item_Visibility': item_visibility,
    'Item_MRP': item_mrp,
    'Item_Outlet_Sales': sales
})

df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'Low Fat': 0, 'Regular': 1})
X = df[['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_MRP']]
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
plt.scatter(y_test, y_pred)
plt.xlabel('True Sales')
plt.ylabel('Predicted Sales')
plt.title('Sales Forecasting Model')
plt.show()
