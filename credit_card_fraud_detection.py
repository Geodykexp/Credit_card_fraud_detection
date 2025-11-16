
import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


import warnings
warnings.filterwarnings('ignore')


# In[163]:


from lightgbm import LGBMClassifier

# Set some visualization styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# In[164]:


df = pd.read_csv("creditcard.csv")


# In[165]:


df.head()


# In[166]:


(df[df["Class"] == 1].iloc[1]).to_json()



df.head()


# In[169]:


df.tail()


# In[170]:


df.shape


# In[171]:


df.info()


# In[172]:


df.isnull().sum().sum()


# In[173]:


from sklearn.preprocessing import StandardScaler


# In[174]:


sc = StandardScaler()
df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))


# In[175]:


df = df.drop(['Time'], axis = 1)


# In[176]:


df.head()


# In[177]:


df.duplicated().any()


# In[178]:


df = df.drop_duplicates()


# In[179]:


df.shape


# In[180]:


df.describe()


# In[181]:


# Check the distribution of the target variable 'Class'
df['Class'].value_counts()


# In[182]:


df.head()


# Data is very Imbalanced.
# Let's visualize it
class_counts = df['Class'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(class_counts.index, class_counts.values)
plt.title('Count of Classes (Matplotlib Bar Plot)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[185]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)


# In[186]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)




# In[188]:


X = df.drop('Class', axis = 1)
X_columns = X.columns.tolist()
y = df['Class']

# In[190]:


y_train = df_train['Class'].values.ravel() # Converts to a 1D numpy array
y_val = df_val['Class'].values.ravel()
y_test = df_test['Class'].values.ravel()


# # One hot encoding

# In[191]:


from sklearn.feature_extraction import DictVectorizer


# In[192]:


dv = DictVectorizer(sparse = False)


# In[

train_dicts = df_train.drop(columns=['Class']).to_dict(orient='records')
val_dicts = df_val.drop(columns=['Class']).to_dict(orient='records')
test_dicts = df_test.drop(columns=['Class']).to_dict(orient='records') 


# In[194]:


X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)
# In[198]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    auc
)


# In[199]:


classifier = {
    "logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
# deleted conditions for 3 models:
# random_state=1
# max_depth=10, random_state=1
# n_estimators=10, random_state=1, n_jobs=-1


# In[200]:


for name, clf in classifier.items():
    print(f'\n==========={name}===========')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f'\n Accuracy: {accuracy_score(y_val, y_pred)}')
    print(f'\n classification_report: {classification_report(y_val, y_pred)}')
    print(f'\n confusion_matrix: {confusion_matrix(y_val, y_pred)}')
    print(f'\n roc_auc_score: {roc_auc_score(y_val, y_pred)}')




# # Another Comparison

# In[201]:


classifier2 = {
    "logistic Regression": LogisticRegression(random_state=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=1, n_jobs=-1)
}
# deleted conditions for 3 models:
# random_state=1
# max_depth=10, random_state=1
# n_estimators=10, random_state=1, n_jobs=-1


# In[202]:


for name, clf in classifier2.items():
    print(f'\n==========={name}===========')
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_test)
    print(f'\n Accuracy: {accuracy_score(y_test, y_pred1)}')
    print(f'\n classification_report: {classification_report(y_test, y_pred1)}')
    print(f'\n confusion_matrix: {confusion_matrix(y_test, y_pred1)}')
    print(f'\n roc_auc_score: {roc_auc_score(y_test, y_pred1)}')


# # DECISION TREE MODEL

# In[204]:


dtc = DecisionTreeClassifier(max_depth = 10)
dtc.fit(X_train, y_train)


# In[205]:


y_pred_val = dtc.predict(X_val)
y_pred_test = dtc.predict(X_test)



# In[207]:


cm = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy:.4f}")
print(confusion_matrix(y_test, y_pred_test))  
print(classification_report(y_test, y_pred_test))


# # RANDOM FOREST MODEL

# In[212]:


rf = RandomForestClassifier(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)


# In[213]:


y_pred_val2 = rf.predict(X_val)
y_pred_test2 = rf.predict(X_test)


# In[215]:


cm = confusion_matrix(y_val, y_pred_val2)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

accuracy = accuracy_score(y_test, y_pred_test2)
print(f"Model Accuracy: {accuracy:.4f}")
print(confusion_matrix(y_test, y_pred_test2))  
print(classification_report(y_test, y_pred_test2))


### Create a Pickle file using serialization 
import pickle
with open('credit_card_fraud_detection.pkl', 'wb') as f: 
    pickle.dump(dv, f) 
    pickle.dump(rf, f)


# Validating PKL file

# In[219]:


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




