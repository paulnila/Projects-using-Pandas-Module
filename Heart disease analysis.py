
# coding: utf-8

# # 1. Import The Libraries And Dataset

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D:\Downloads\heart.csv')


# # 2. Display Top 5 Rows of The Dataset

# In[3]:


df.head(5)


# # 3. Check The Last 5 Rows of The Dataset

# In[4]:


df.tail(5)


# # 4. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[8]:


print("Rows: ",df.shape[0])
print("Columns: ",df.shape[1])


# # 5. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[10]:


df.info()


# # 6. Check Null Values In The Dataset

# In[12]:


df.isnull().sum()


# # 7. Check For Duplicate Data and Drop Them

# In[5]:


data_duplicate = df.duplicated().any()
print(data_duplicate)


# In[6]:


data=df.drop_duplicates()
print(data)


# # 8. Get Overall Statistics About The Dataset

# In[19]:


df.describe()


# In[21]:


df['target']


# # 9. Draw Correlation Matrix 

# In[34]:


plt.figure(figsize=(17,6))
sns.heatmap(df.corr(),annot = True)


# # 10. How Many People Have Heart Disease, And How Many Don't Have Heart Disease In This Dataset?

# In[25]:


df.columns


# In[43]:


data['target'].value_counts()


# In[45]:


sns.countplot(data['target'])


# # 11. Find Count of  Male & Female in this Dataset
# 

# In[5]:


df.columns


# In[9]:


data['sex'].value_counts()


# In[24]:


sns.countplot(data['sex'])
plt.xticks([0,1],['Female','Male'])
plt.show()


# # 12. Find Gender Distribution According to The Target Variable

# In[13]:


df.columns


# In[28]:


sns.countplot(x='sex',hue="target",data=data)
plt.xticks([0,1],['Female','Male'])
plt.legend(labels = ['No-disease','disease'])


# In[29]:


df['age']


# # 13. Check Age Distribution In The Dataset

# In[36]:


sns.distplot(data['age'],bins=20)


# In[37]:


data.columns


# # 14. Check Chest Pain Type

# In[46]:


sns.countplot(data['cp'])
plt.xticks([0,1,2,3],["typical angina","atypical angina","non-anginal pain","asymptomatic"])
plt.xticks(rotation = 60)


# # 15. Show The Chest Pain Distribution As Per Target Variable

# In[57]:



sns.countplot(x=data['cp'],hue=data['target'])

plt.xticks(rotation = 0)
plt.legend(labels=["No-disease","Disease"])


# # 16. Show Fasting Blood Sugar Distribution According To Target Variable

# In[58]:


df.columns


# In[59]:


data['fbs']


# In[64]:


sns.countplot(x='fbs',hue='target',data=data)
plt.legend(["No disease","disease"])


# # 17.  Check Resting Blood Pressure Distribution

# In[65]:


sns.distplot(data['trestbps'])


# In[66]:


data['trestbps'].hist()


# # 18. Compare Resting Blood Pressure As Per Sex Column

# In[67]:


df.columns


# In[76]:


g = sns.FacetGrid(data, hue = "sex",aspect = 4)                
g.map(sns.kdeplot,'trestbps',shade = True)
plt.legend(labels=['Male','Female'])


# # 19. Show Distribution of Serum cholesterol

# In[77]:


data['chol'].hist()


# # 20. Plot Continuous Variables

# In[1]:


category_columns =[]
cont_columns = []


# In[8]:


# separating categorical and continuous variable

for column in data.columns:
    if data[column].nunique() <= 10:
        category_columns.append(column)
    else:
        cont_columns.append(column)


# In[9]:


cont_columns


# In[16]:


data.hist(cont_columns, figsize = (15,6))
plt.tight_layout()
plt.show()

