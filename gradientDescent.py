import numpy as np
from numpy.core.numeric import ones
import pandas as pd
import matplotlib.pyplot as plt

data = "data.txt"

def costFunction(X, y, theta):
    m = X.shape[0]
    predict = np.dot(X, theta)
    res = 1/(2*m)*np.sum((predict-y)**2)
    return res  

def gradientDescent(X, y, theta, alpha, num_iters):
    cost_hist = []
    for i in range(num_iters):
        temp = (np.dot(X, theta)-y)
        theta = theta-(alpha*(np.dot(temp, X)/m))
        cost_hist.append(costFunction(X, y, theta))
    return theta, cost_hist

df = pd.read_csv(data, header = None)
m = len(df.index) 
X= np.column_stack((np.ones((m,1)), np.array(df.iloc[:,0].to_numpy().reshape(m,1))))
y = df.iloc[:,1].to_numpy()
theta = np.zeros(2)
alpha = 0.01
num_iters = 1500

theta, costs = gradientDescent(X, y, theta, alpha, num_iters)
print("theta found by gradiant descent after", num_iters, "iterations:", theta[0], " ", theta[1])
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

predict1 = np.array([1,3.5])
result1 = np.dot(predict1, theta)
print("For value:", predict1[1], "we predict:", result1)
