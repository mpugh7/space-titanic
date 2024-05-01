#!/usr/bin/env python
# coding: utf-8

# The goal of this analysis is to predict whether a given passenger on the Space Ship Titanic was going to be teleported to a different dimension. Before any of this analysis could be done, a lot of feature engineering had to be performed on the data in order to make the analysis as effective as possible. After changing many of the features into numeric values, I was able to use a Random Forest Classifier in order to first predict the "Transported" values for the training and testing sets given to me by Kaggle, and then predict those same values for part of our training set. This secondary analysis part was added in order to be able to quantify the accuracy of our model.

# In[310]:


import pandas as pd
import matplotlib.pyplot as plt
test=pd.read_csv(r"C:\Users\Owner\Downloads\test.csv")
train=pd.read_csv(r"C:\Users\Owner\Downloads\train.csv")
passid=pd.read_csv(r"C:\Users\Owner\Downloads\sample_submission.csv")
test=test.dropna()
train=train.dropna()
passid


# In[311]:


train["CryoSleep"] = train["CryoSleep"].astype(int)
train['VIP']=train['VIP'].astype(int)
test["CryoSleep"] = test["CryoSleep"].astype(int)
test['VIP']=test['VIP'].astype(int)
train['Transported']=train['Transported'].astype(int)
train['TOT$']=train['RoomService']+train['FoodCourt']+train['ShoppingMall']+train['Spa']+train['VRDeck']
test['TOT$']=test['RoomService']+test['FoodCourt']+test['ShoppingMall']+test['Spa']+test['VRDeck']
train


# The above lines convert the "CryoSleep", "VIP", and "Transported" variables from Boolean values into integers so that they can be used in the analysis later. The last few lines in the above code create a new feature called "TOT$" which adds up all of the money that each individual passenger uses and displays it in a new features.

# The plot below was created with the goal of seeing if there was a corellation between age and money spent with whether a passenger was transported or not, and based on the plot, no such corellation exists

# In[312]:


plt.scatter(train['Age'],train['TOT$'],c=train['Transported'],label='Transported')
plt.legend(loc='upper right')


# In[313]:


new_list = []
for i in test['HomePlanet'].values:
    if i=='Earth':
        new_list.append(1)
    elif i=='Mars':
        new_list.append(2)
    elif i=='Europa':
        new_list.append(3)
test['HomePlanet']=new_list

new_list = []
for i in train['HomePlanet'].values:
    if i=='Earth':
        new_list.append(1)
    elif i=='Mars':
        new_list.append(2)
    elif i=='Europa':
        new_list.append(3)
train['HomePlanet']=new_list


# In[314]:


new_list = []
for i in test['Destination'].values:
    if i=='TRAPPIST-1e':
        new_list.append(1)
    elif i=='55 Cancri e':
        new_list.append(2)
    elif i=='PSO J318.5-22':
        new_list.append(3)
test['Destination']=new_list

new_list = []
for i in train['Destination'].values:
    if i=='TRAPPIST-1e':
        new_list.append(1)
    elif i=='55 Cancri e':
        new_list.append(2)
    elif i=='PSO J318.5-22':
        new_list.append(3)
train['Destination']=new_list


# The above two for loops were created with the purpose of assigning numerical values to the values of the "Destination" and
# "HomePlanet" variables. This process was done by iterating through each of the columns values, and then checking if each value was equal to a certain planet, and it if was, that planets numerical value was added to a list. At the end, each column was then overwritten by the lists of numbers that were created in the for loops.

# In[315]:


new_list = []
for i in test['Cabin'].values:
    if '/S' in i:
        new_list.append(1)
    elif '/P' in i:
        new_list.append(2)
test['Boat Side']=new_list

new_list = []
for i in train['Cabin'].values:
    if '/S' in i:
        new_list.append(1)
    elif '/P' in i:
        new_list.append(2)
train['Boat Side']=new_list


# In[316]:


new_list = []
levels=['A','B','C','D','E','F','G','T']
for i in test['Cabin'].values:
    count=0
    for j in levels:
        count+=1
        if j+'/' in i:
            new_list.append(count)
test['Boat Level']=new_list

new_list=[]
for i in train['Cabin'].values:
    count=0
    for j in levels:
        count+=1
        if j+'/' in i:
            new_list.append(count)
train['Boat Level']=new_list


# The above to for loops are similar to the other two we have discussed, bu with minor differences. The first loop is iterating through the Cabin column in order to check if a passenger is on the Starboard or Port sides of the ship. This is done through checking if the string of "/S" or "/P" is included in the "Cabin" value for each passenger. If it is, the list method mentioned about the for loops mentioned above is employed. The second loop actually includes a nested for loop, as we are not only interating through the list of "Cabin" values, but also the "levels" values. This is done to check if the levels values when added to the front of the '/' string is included in the Cabin values to see what level of the ship each passenger is on.

# Homeplanet 1 is Earth.
# Homeplanet 2 is Mars.
# Homeplanet 3 is Europa.
# Destination 1 is TRAPPIST-1e.
# Destination 2 is 55 Cancri e.
# Destination 3 is PSO J318.5-22.
# For Boat Side, Starboard is 1 and Portside is 2.
# For Boat Level, the levels are labeled 1-8 in the order of ['A','B','C','D','E','F','G','T'].

# In[317]:


train


# Below are lines of code that calculate the corelation values for each numerical value, and then spit them out into corelation matrix' and heat maps. The larger or smaller the corelation values are, the more positively or negatively corelated the variable are respectively.

# In[318]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

corr = train.corr()

sns.heatmap(corr)
train.corr()


# In[319]:


X_test=test[['Spa','RoomService','VRDeck','CryoSleep']]
y_test=passid['Transported']
ftest=test[['Spa','RoomService','VRDeck','CryoSleep','PassengerId']]
X_train=train[['Spa','RoomService','VRDeck','CryoSleep']]
y_train=train['Transported']


# In[320]:


from sklearn.ensemble import RandomForestClassifier 
  
randomforest = RandomForestClassifier() 
  
randomforest.fit(X_train, y_train) 


# Random forest classification is essentially a decision tree in which the model is trained upon the trees of the training data, and then the results of those trees are used to predict the outcomes for the testing set. In this instance, we were able to predict the Transported column of our testing set based upon what the corelation matrix said were the variables most corelated with the "Transported" variable.

# In[321]:


ids = ftest['PassengerId']
predictions = randomforest.predict(ftest.drop('PassengerId', axis=1)) 
  
output = pd.DataFrame({'PassengerId': ids, 'Transported': predictions})
output


# In[322]:


predictors=train[['Spa','RoomService','VRDeck','CryoSleep']]
target = train["Transported"] 
x_train, x_test, y_train, y_test = train_test_split( 
    predictors, target, test_size=0.2, random_state=0)


# In[323]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
  
randomforest = RandomForestClassifier() 
   
randomforest.fit(x_train, y_train) 
y_pred = randomforest.predict(x_test) 
  
ac = round(accuracy_score(y_pred, y_test) * 100, 2) 
print(ac)http://localhost:8892/notebooks/DSC%20440%20Project%202.ipynb#


# The final piece of analysis performed here was checking the accuracy of our forest classifier by splitting our training set into smaller training and testing sets. More specifically the training set was 80% of our data and the testing set was the other 20%. A "random_state" was also chosen when splitting our data so that the dataset would be split the same each time the code is run, and thus our analysis can be replicated. An accuracy score was then calculated by comparing our predicted "Transported" values and our actual "Transported" values. Overall, the best accuracy for our classifier was found using the Spa, Room Service, VRDeck, and CryoSleep variables and this accuracy was around 74%. This shows how even our best model for this analysis was not that accurate and thus it can be concluded that although these variables are our best predictors, they are still not very good at predicting whether a passenger on the Space Ship Titanic was transported to another dimension.
