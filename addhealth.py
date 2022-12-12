

import pandas as pd
# Reading data from csv
df = pd.read_csv("addhealth.csv")
# Removing NaN values with Most Frequent value of the column
for i in df:
    df[i] = df[i].fillna(df[i].mode()[0])



######## USING LOGISTIC REGRESSION ########

######## Solution for Part 1 ########
'''Build a classification model evaluating if an adolescent would smoke 
regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, 
Black, Native American and Asian, alcohol use, alcohol problems, marijuana use,
cocaine use, inhalant use, availability of cigarettes in the home, depression,
and self-esteem.'''

# Separating data into Independent and Dependent Variables
# Separating Dependent and Independent variables as per Problem Statement
fe = df[['BIO_SEX','age','WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN',
           'ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail',
           'DEP1','ESTEEM1']].values
la = df["TREG1"].values

def LRModel(features, labels):
    
    # Splitting the dataset into train and test
    from sklearn.model_selection import train_test_split as TTS
    
    f_train,f_test,l_train,l_test = TTS(fe, la, test_size = 0.25,
                                        random_state = 0)
    
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0)
    reg = reg.fit(f_train, l_train)
    
    pred = reg.predict(f_test)   # Prediction on test data   
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(l_test, pred)
    
    # check the accuracy on the Model
    mod_score = reg.score(f_test, l_test)
    
    return pred,cm,mod_score

Pred, CM, Score = LRModel(fe,la)

print ("model accuracy using confusion matrix (LogisticRegression): "+str(CM))
print ("model accuracy using .score() function (LogisticRegression): "+str(round(Score*100,2)))


######## Solution for Part 2 ########
#an adolescent gets expelled or not from school


'''Build a classification model evaluation if an adolescent gets expelled or
 not from school based on their Gender and violent behavior.'''


fe1 = df[["BIO_SEX","VIOL1"]].values
la1 = df["EXPEL1"].values

def LRModel1(features, labels):
    
    # Splitting the dataset into train and test
    from sklearn.model_selection import train_test_split as TTS
    
    f_train,f_test,l_train,l_test = TTS(fe1, la1, test_size = 0.25,
                                        random_state = 0)
    
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0)
    reg = reg.fit(f_train, l_train)
    
    Pred1 = reg.predict(f_test)   # Prediction on test data   
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    CM1 = confusion_matrix(l_test, Pred1)
    
    # check the accuracy on the Model
    Score1 = reg.score(f_test, l_test)
    
    return Pred1,CM1,Score1

Pred1, CM1, Score1 = LRModel1(fe1,la1)

print ("model accuracy using confusion matrix (LogisticRegression): "+str(CM1))
print ("model accuracy using .score() function (LogisticRegression): "+str(round(Score1*100,2)))


######## Solution for Part 3 ########


'''Build a classification model in relation to regular smokers as a target and explanatory 
variable specifically with Hispanic, White, Black, Native American and Asian.'''

fe2 = df[['WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN']].values
la2 = df["TREG1"].values

def LRModel2(features, labels):
    
    # Splitting the dataset into train and test
    from sklearn.model_selection import train_test_split as TTS
    
    f_train,f_test,l_train,l_test = TTS(fe2, la2, test_size = 0.25,
                                        random_state = 0)
    
    # Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0)
    reg = reg.fit(f_train, l_train)
    
    Pred2 = reg.predict(f_test)   # Prediction on test data   
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    CM2 = confusion_matrix(l_test, Pred2)
    
    # check the accuracy on the Model
    Score2 = reg.score(f_test, l_test)
    
    return Pred2,CM2,Score2

Pred2, CM2, Score2 = LRModel2(fe2,la2)

print ("model accuracy using confusion matrix (LogisticRegression): "+str(CM2))
print ("model accuracy using .score() function (LogisticRegression): "+str(round(Score2*100,2)))




######## USING KNN ########


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix, accuracy_score

# Applying KNN Classifier
classi = KNeighborsClassifier(n_neighbors = 8)


######## Solution for Part 1 ########

# Separating Dependent and Independent variables as per Problem Statement
fe = df[['BIO_SEX','age','WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN',
           'ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail',
           'DEP1','ESTEEM1']].values
la = df["TREG1"].values

# Splitting the Data into Test and Train
ftr,fte,ltr,lte = TTS(fe,la,test_size=.2,random_state=0)


classi.fit(ftr,ltr)
pred = classi.predict(fte)

# Building Confusion Matrix
CM = confusion_matrix(pred,lte)

# Getting Accuracy Score of the Model
Score = accuracy_score(lte,pred)
print ("model accuracy using confusion matrix (KNN): "+str(CM))
print ("model accuracy using .score() function (KNN): "+str(round(Score*100,2))+"%")



######## Solution for Part 2 ########

# Separating Dependent and Independent variables as per Problem Statement
fe1 = df[["BIO_SEX","VIOL1"]].values
la1 = df["EXPEL1"].values

# Splitting the Data into Test and Train
ftr,fte,ltr,lte = TTS(fe1,la1,test_size=.2,random_state=0)


classi.fit(ftr,ltr)
Pred1 = classi.predict(fte)

# Building Confusion Matrix
CM1 = confusion_matrix(Pred1,lte)

# Getting Accuracy Score of the Model
Score1 = accuracy_score(lte,Pred1)

print ("model accuracy using confusion matrix (KNN): "+str(CM1))
print ("model accuracy using .score() function (KNN): "+str(round(Score1*100,2))+"%")


######## Solution for Part 3 ########

# Separating Dependent and Independent variables as per Problem Statement
fe2 = df[['WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN']].values
la2 = df["TREG1"].values

# Splitting the Data into Test and Train
ftr,fte,ltr,lte = TTS(fe2,la2,test_size=.2,random_state=0)


classi.fit(ftr,ltr)
Pred2 = classi.predict(fte)

# Building Confusion Matrix
CM2 = confusion_matrix(Pred2,lte)

# Getting Accuracy Score of the Model
Score2 = accuracy_score(lte,Pred2)

print ("model accuracy using confusion matrix (KNN): "+str(CM2))
print ("model accuracy using .score() function (KNN): "+str(round(Score2*100,2))+"%")