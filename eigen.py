#libraries
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

df_sensor_data = pd.read_csv('HT_Sensor_dataset.dat', delim_whitespace=True)
del df_sensor_data['id']
del df_sensor_data['time']
del df_sensor_data['Temp.']
del df_sensor_data['Humidity']
#print(df_sensor_data.head())
print(df_sensor_data.shape)
#print(df_sensor_data.dtypes)
#print(df_sensor_data.isna)


df = df_sensor_data.notna().astype('float64')

covariance_matrix = df_sensor_data.cov()
numpy_covariance_matrix = covariance_matrix.values
print(covariance_matrix)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
#plt.imshow(numpy_covariance_matrix)
#plt.show()
#plt.close()

#Transformation Matrix

eVe, eVa = np.linalg.eig(covariance_matrix)

plt.scatter(covariance_matrix[:, 0], covariance_matrix[:, 1])
for e, v in zip(eVe, eVa.T):
    plt.plot([0, 3*np.sqrt(e)*v[0]], [0, 3*np.sqrt(e)*v[1]], 'k-', lw=2)
plt.title('Transformed Data')
plt.axis('equal');
plt.show()

values, vectors = eig(covariance_matrix)
#print(values)
#print(vectors)





































