###########################################################
### Exercise 0: Data Exploration and Data Cleaning
###########################################################
import pandas as pd

datafile= 'E:/Learn_Python/flight_customer/data/preprocesseddata.xls' 
cleanedfile = 'E:/Learn_Python/flight_customer/data/data_cleaned.xls'

data = pd.read_excel(datafile) 
print data.iloc[range(5)] # check read-in data

explore = data.describe(percentiles = [], include = 'all').T
explore['null'] = len(data)-explore['count']

explore = explore[['null','max','min']]
explore

data = data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()]
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)
data = data[index1 | index2 | index3]
data.to_excel(cleanedfile)

###########################################################
### Exercise 1: Z-score Standardization
###########################################################
import pandas as pd
inputfile = 'E:/Learn_Python/flight_customer/data/zscoredata.xls'
outputfile = 'E:/Learn_Python/flight_customer/data/zscore_std_data.xls' 

data_raw = pd.read_excel(inputfile,header=0) #first row used as header
print data_raw.iloc[range(5)] # check read-in data

data_std = (data_raw - data_raw.mean(axis = 0))/(data_raw.std(axis = 0)) # axis=0 for vertical agg functions
data_std.columns = ['Z_'+i for i in data_std.columns] # rename the headers after standardization

print data_std.iloc[range(5)] 
data_std.to_excel(outputfile,index=False) # output data to excel file
 
###########################################################
### Exercise 2: KMeans Clustering
###########################################################
import pandas as pd
from sklearn.cluster import KMeans

inputfile = 'E:/Learn_Python/flight_customer/data/zscore_std_data.xls'
k=5 # number of clusters

data_std = pd.read_excel(inputfile,header=0) #first row used as header
print data_std.iloc[range(5)] 

kmodel = KMeans(n_clusters=k, n_jobs=4)
kmodel.fit(data_std)
kmodel.cluster_centers_
kmodel.labels_
pd.Series(kmodel.labels_).value_counts() # cluster count

###########################################################
### Exercise 3: Cluster plot (radar chart)
###########################################################
import numpy as np
import matplotlib.pyplot as plt

labels = data_std.columns 
k = 5
plot_data = kmodel.cluster_centers_
color = ['b', 'g', 'r', 'c', 'y'] 

plot_data = kmodel.cluster_centers_

angles = np.linspace(0, 2*np.pi, k, endpoint=False)
angles = np.concatenate((angles, [angles[0]])) 
plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1) 

fig = plt.figure()
ax = fig.add_subplot(111, polar=True) 
ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5))
ax.set_thetagrids(angles * 180/np.pi, labels)
plt.legend(loc = 4)

for i in range(len(plot_data)):
    ax.plot(angles, plot_data[i], 'o-', color = color[i], label = 'cluster'+str(i), linewidth=2)

plt.show()

