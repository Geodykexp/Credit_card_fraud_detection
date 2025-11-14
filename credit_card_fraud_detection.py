import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


from lightgbm import LGBMClassifier

# Set some visualization styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# In[4]:


df = pd.read_csv("creditcard.csv")


# In[5]:


df.head()


# In[6]:


pd.options.display.max_columns = None


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.isnull().sum().sum()


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()
df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))


# In[14]:


df = df.drop(['Time'], axis = 1)


# In[15]:


df.head()


# In[16]:


df.duplicated().any()


# In[17]:


df = df.drop_duplicates()


# In[18]:


df.shape


# In[19]:


df.describe()


# In[20]:


# Check the distribution of the target variable 'Class'
df['Class'].value_counts()


# In[21]:


df.head()


# In[22]:


# Data is very Imbalanced.
# Let's visualize it
class_counts = df['Class'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(class_counts.index, class_counts.values)
plt.title('Count of Classes (Matplotlib Bar Plot)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# In[24]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)


# In[25]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)


# In[26]:


len(df_train), len(df_val), len(df_test)


# In[27]:


X = df.drop('Class', axis = 1)
X_columns = X.columns.tolist()
y = df['Class']


# In[28]:


train_dicts = df_train[X_columns].to_dict(orient = 'records')
val_dicts = df_val[X_columns].to_dict(orient = 'records')


# In[29]:


y_train = df_train['Class'].values.ravel() # Converts to a 1D numpy array
y_val = df_val['Class'].values.ravel()
y_test = df_test['Class'].values.ravel()


# # One hot encoding

# In[30]:


from sklearn.feature_extraction import DictVectorizer


# In[31]:


dv = DictVectorizer(sparse = False)


# In[32]:


train_dicts = df_train.to_dict(orient = 'records')
val_dicts = df_val.to_dict(orient = 'records')
test_dicts = df_test.to_dict(orient = 'records')


# In[33]:


X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)


# In[34]:


X_train


# In[35]:


X_val


# In[36]:


X_test


# In[ ]:





# # Modelling

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    
)
import xgboost as xgb


# In[38]:


classifier = {
    "logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
# deleted conditions for 3 models:
# random_state=1
# max_depth=10, random_state=1
# n_estimators=10, random_state=1, n_jobs=-1


# In[39]:


for name, clf in classifier.items():
    print(f'\n==========={name}===========')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f'\n Accuracy: {accuracy_score(y_val, y_pred)}')
    print(f'\n classification_report: {classification_report(y_val, y_pred)}')
    print(f'\n confusion_matrix: {confusion_matrix(y_val, y_pred)}')
    print(f'\n roc_auc_score: {roc_auc_score(y_val, y_pred)}')



# Since decision tree and random forest have the same accuracy, 
# This project will go with the decision tree model for simplicity.


# In[ ]:

# # DECISION TREE MODEL

# In[41]:


dtc = DecisionTreeClassifier(max_depth = 10)
dtc.fit(X_train, y_train)


# In[42]:


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()



y_pred = dtc.predict(X_test)  


# In[44]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))



### Create a Pickle file using serialization 
import pickle
with open('credit_card_fraud_detection.pkl', 'wb') as f:
    dv = pickle.load(f)
    dtc = pickle.load(f)

# pickle_out = open("credit_card_fraud_detection.pkl","wb")
# pickle.dump(dv, pickle_out)
# pickle.dump(dtc, pickle_out)
# pickle_out.close()


# In[ ]:





# # Validating PKL file

# In[46]:


import pickle

with open('credit_card_fraud_detection.pkl', 'rb') as f:
    loaded_object = pickle.load(f)

print(f"Type of loaded object: {type(loaded_object)}")

# If you think it should be a DataFrame
try:
    print(f"Shape of DataFrame: {loaded_object.shape}")
except AttributeError:
    pass # Not a DataFrame

# If you think it should be a Decision Tree model
try:
    print(f"Model coefficients shape: {loaded_object.coef_.shape}")
except AttributeError:
    pass # Not a DecisionTreeClassifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




