import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./winequality-white.csv",sep=";")

train_data=data[:1000]
data_X=train_data.iloc[:,0:11]
data_Y=train_data.iloc[:,11:12]
#print(train_data.columns)
print(data_X)
print(data_Y)

# Linha 1: CORRETA (mas tem gap)
evaluation_data = data[1000:]  # Funciona, mas pula a linha 1000

# Linha 2: CORRETA
data_X = evaluation_data.iloc[:, 0:11]  # Seleciona primeiras 11 colunas

# Linha 3: CORRETA
data_Y = evaluation_data.iloc[:, 11:12]  # Seleciona coluna 12 como DataFrame

# Linha 4: CORRETA
print(type(evaluation_data))  # Mostra o tipo

# Linha 5: CORRETA
print(type(data_X))  # Mostra o tipo

loaded_model = p1.load(open('white-wine_quality_predictor', 'rb'))

# Linha 7: PROBLEMA DE DIGITAÇÃO - "Coefficientes" (português) vs "Coefficients" (inglês)
print("Coeficientes: \n", loaded_model.coef_)  # Funciona, mas texto estranho

# Linha 8: CORRETA
y_pred = loaded_model.predict(data_X)

# Linha 9: PROBLEMA - shape incompatível
z_pred = y_pred - data_Y  # PODE DAR ERRO!

from sklearn.metrics import accuracy_score, mean_absolute_error

# Converter predições
y_pred_int = np.round(y_pred).astype(int).flatten()
y_true = data_Y.values.flatten().astype(int)

# Calcular acertos
right = np.sum(y_pred_int == y_true)
wrong = np.sum(y_pred_int != y_true)
total = len(y_true)

# Métricas
accuracy = (right / total) * 100
mae = mean_absolute_error(y_true, y_pred_int)

print(f"Total de amostras: {total}")
print(f"Acertos: {right} ({accuracy:.2f}%)")
print(f"Erros: {wrong} ({100-accuracy:.2f}%)")
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")