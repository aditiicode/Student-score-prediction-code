# Student Score Prediction using Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (Hours studied vs Scores)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
scores = np.array([35, 40, 50, 55, 65, 70, 75, 85, 90])

# Create model
model = LinearRegression()
model.fit(hours, scores)

# Prediction
predicted_score = model.predict([[7]])
print("Predicted score for 7 hours study:", predicted_score[0])

# Plot graph
plt.scatter(hours, scores)
plt.plot(hours, model.predict(hours))
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study Hours vs Score Prediction")
plt.show()
