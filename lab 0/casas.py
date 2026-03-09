import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


#XX = np.array([[60, 98400], [70, 99100], ...,[250, 352500],[260, 457600]])
XX = pd.read_csv("C:/Users/alves/Documents/GitHub/Carlasofiaaautom-tica/lab 0/casas1.csv")
X=np.array(XX, ndmin=2) #matriz n\times2
y=X[:, 1:].T
y=y[0]
X=X[:, :1]
print(y)
print(X)
regr = linear_model.LinearRegression()
z = regr.fit(X, y)
yz = regr.predict(X)
yz = yz.round(3)

plt.title("Preços das casas vs Área em metros quadrados")
plt.xlabel('Área em ($m^2$)')
plt.ylabel('Preço Estimado (¿)')
plt.scatter(X, y, color="black")
plt.plot(X, yz, color="blue", linewidth=3)
plt.show()

print(y)
print(yz)
print(regr.predict([[300]]).round(3))
