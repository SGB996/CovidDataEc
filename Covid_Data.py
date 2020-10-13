import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Datos historicos contagios totales Covid-19 Ecuador
path1="data/Covid_dataset.csv"
path2='data/Datasetcov-19.csv'
df=pd.read_csv(path1)
#describe=df.describe()
#print(describe)
y=df['c_totales']
x=df['fecha']
""" plt.figure(figsize=(12, 10))
plt.plot(x,y)
plt.xlabel('Fecha')
plt.ylabel('Casos Totales')
plt.title('Casos Totales Covid-19 Ecuador desde\n 29 de Febrero hasta 16 de Agosto')
plt.show()
 """
#DatasetCov-19 agrupado por meses y dias final desde 01-06-2020 hasta 16-08-2020
pathf='data/Finaldataset.csv'
df_2=pd.read_csv(pathf)
junAug=df_2.loc[df_2['mes']>=6]
time=junAug[['index']]
xplot=junAug['index']
casos=junAug['c_totales']

#Data set split using Train Test split function  
X_train, X_test, y_train, y_test = train_test_split(time, casos, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Entrenamiento para predecir casos de coronavirus usando regresion lineal con el dataset actual
lm=LinearRegression()
lm.fit(X_train, y_train)
print('el intercepto: ', lm.intercept_)
print('el coeficiente: ', lm.coef_)

#Prediction using the test set

yhat=lm.predict(X_test)

#Model evaluation using MSE, and R-squared
r2 = r2_score(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
print('R^2 is: ', r2)
print('Mean squaed error is: ', mse)

#grafica de prediccion
plt.figure(figsize=(12, 10))
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, lm.coef_*X_train + lm.intercept_, 'r')
plt.xlabel('Meses: Junio-Agosto')
plt.ylabel('Casos Totales Junio-Agosto')
plt.title('Prediccion Casos Totales Covid-19 Ecuador\n meses de Junio hasta Agosto')
plt.show()


