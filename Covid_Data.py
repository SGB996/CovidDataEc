import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
#Datos historicos contagios totales Covid-19 Ecuador
path1="Covid_dataset.csv"
path2='Datasetcov-19.csv'
df=pd.read_csv(path1)
#describe=df.describe()
#print(describe)
y=df['c_totales']
x=df['fecha']
plt.figure(figsize=(12, 10))
plt.plot(x,y)
plt.xlabel('Fecha')
plt.ylabel('Casos Totales')
plt.title('Casos Totales Covid-19 Ecuador desde\n 29 de Febrero hasta 16 de Agosto')
plt.show()

#DatasetCov-19 agrupado por meses y dias final desde 01-06-2020 hasta 16-08-2020
pathf='Finaldataset.csv'
df_2=pd.read_csv(pathf)
junAug=df_2.loc[df_2['mes']>=6]
time=junAug[['index']]
xplot=junAug['index']
casos=junAug['c_totales']

#Entrenamiento para predecir casos de coronavirus usando regresion lineal con el dataset actual
lm=LinearRegression()
lm.fit(time,casos)
print('el intercepto es: ',lm.intercept_)
print('la pendiente es: ',lm.coef_)


#Calculo error R^2
error=lm.score(time,casos)
print('R^2 es: ',error)

#Prediccion
test=np.arange(1,101,1).reshape(-1,1)
prediction=lm.predict(test)
print('la prediccion es: ',prediction[53])

#grafica de prediccion
plt.figure(figsize=(12, 10))
sns.regplot(x=xplot, y=casos, data=junAug, marker='+')
plt.xlabel('Meses: Junio-Agosto')
plt.ylabel('Casos Totales Junio-Agosto')
plt.title('Casos Totales Covid-19 Ecuador\n meses de Junio hasta Agosto')
plt.show()


