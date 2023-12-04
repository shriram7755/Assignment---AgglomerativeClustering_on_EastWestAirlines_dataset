# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:45:05 2023

@author: SHRI
"""

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

#now import file  from data set and create a dataframe

df=pd.read_excel('EastWestAirlines3.xlsx')

#finding data type of columns
df.dtypes
'''
ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64
'''

#finding out 5 number summary of dataframe
df.describe()
'''
Out[60]: 
               ID#       Balance  ...  Days_since_enroll       Award?
count  3999.000000  3.999000e+03  ...         3999.00000  3999.000000
mean   2014.819455  7.360133e+04  ...         4118.55939     0.370343
std    1160.764358  1.007757e+05  ...         2065.13454     0.482957
min       1.000000  0.000000e+00  ...            2.00000     0.000000
25%    1010.500000  1.852750e+04  ...         2330.00000     0.000000
50%    2016.000000  4.309700e+04  ...         4096.00000     0.000000
75%    3020.500000  9.240400e+04  ...         5790.50000     1.000000
max    4021.000000  1.704838e+06  ...         8296.00000     1.000000

[8 rows x 12 columns]
'''


#find no of columns 
df.columns()
'''
Index(['Customer', 'State', 'Customer Lifetime Value', 'Response', 'Coverage',
       'Education', 'Effective To Date', 'EmploymentStatus', 'Gender',
       'Income', 'Location Code', 'Marital Status', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel', 'Total Claim Amount',
       'Vehicle Class', 'Vehicle Size'],
      dtype='object')
'''

#finding null value

df.isnull().sum()
'''
Out[62]: 
ID#                  0
Balance              0
Qual_miles           0
cc1_miles            0
cc2_miles            0
cc3_miles            0
Bonus_miles          0
Bonus_trans          0
Flight_miles_12mo    0
Flight_trans_12      0
Days_since_enroll    0
Award?               0
dtype: int64
'''

#finding the box plot for multiple columns

sns.boxplot(df['total_bill'], orient="h")



#finding outlier for multiple columns


def detect_outliers_iqr(data):
  
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

   
    IQR = Q3 - Q1
    
  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
   
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers




outliers_column1 = detect_outliers_iqr(column1_values)
outliers_column2 = detect_outliers_iqr(column2_values)





# we know there is scale difference between among the columns,r
#which we have remove
#either by using the normalization and standardization
#when ever there is mixed data apply normalization
ewar.info()


def norm_func(i):
    x=(i-i.min())/(i.max()- i.min())
    return x

#Now apply this normalization function to dataframe
# for all the rows and columns from 1 until end
# Since 0th columns has university names henced skipped
df_norm=norm_func(ewar.iloc[:,1:])
df_norm


b=df_norm.describe()


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering Tendrogram");
plt.xlabel('Index')
plt.ylabel('Distance')


sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show


#TEndogram
#apply agglomerative clustering choosing 3 as clusters
#from Tendogram
#it is just showing the number of possible clusters

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean').fit(df_norm)
#label to the clusters

cluster_label=pd.Series(h_complete.labels_)

ewar['clust']=cluster_label


ewar.to_csv("east_west_air.csv",encoding="utf-8")
import os
os.getcwd()
