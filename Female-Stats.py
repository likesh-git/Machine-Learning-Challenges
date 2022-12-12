


# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Female_Stats.Csv')

# Check data Types for each columns
print(dataset.dtypes)

# Seperate Features and Labels
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:, [0]].values

# Check Column wise is any data is missing or NaN
dataset.isnull().any(axis=0)

# Check data Types for each columns
print(dataset.dtypes)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
# Whether we have Univariate or Multivariate, class is LinearRegression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

Pred = regressor.predict(features_test)

print (pd.DataFrame(zip(Pred, labels_test)))

#eg1 :
x = [75,75]
x = np.array(x)
x = x.reshape(1,2)
Pred1 = regressor.predict(x)





"""

When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.

"""


#Father's height constant
#test1
x = [75,75]
x = np.array(x)
x = x.reshape(1,2)
Pred1 = regressor.predict(x)
print(Pred1)

#test2
x = [76,75]  #mom height increase by one
x = np.array(x)
x = x.reshape(1,2)
Pred2 = regressor.predict(x)
print(Pred2)

#difference
diff = Pred2 - Pred1
print("Student height will be increase by ",diff)   #[[0.30645861]]



"""

When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.



"""


#mothers's height constant
#test1
x = [75,75]
x = np.array(x)
x = x.reshape(1,2)
Pred1 = regressor.predict(x)
print(Pred1)

#test2
x = [75,76]  #mom height increase by one
x = np.array(x)
x = x.reshape(1,2)
Pred2 = regressor.predict(x)
print(Pred2)

#difference
diff = Pred2 - Pred1
print("Student height will be increase by ",diff)   #[[0.40262219]]













# Version 2 of solution 




import pandas as pd
import numpy as np

dataset=pd.read_csv("stats_females.csv")

features=dataset.iloc[:,1:]
labels=dataset.iloc[:,0]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(features_train,labels_train)

import statsmodels.formula.api as sm
features=np.append(arr=np.ones((214,1)).astype(int),values=features,axis=1)

features_opt=features[:,[0,1,2]]
regressor_OLS=sm.OLS(labels,features_opt).fit()
regressor_OLS.summary()

"""
When Father’s Height Is Held Constant, The Average Student Height Increases 
By How Many Inches For Each One-Inch Increase In Mother’s Height.
"""
print("When Father's Height is Held Constant then the average height increase by",regressor_OLS.params[1])

print("When Mother's Height is Held Constant then the average height increase by",regressor_OLS.params[2])