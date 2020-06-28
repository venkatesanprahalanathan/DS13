
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[97]:


df=pd.read_csv("C:\\Users\\venkatesan-p\\Downloads\\datasets_228_482_diabetes.csv")


# In[98]:


df.info()


# In[99]:


df.isnull().sum()#evebthough its non null check for our purpsse


# In[100]:


df.head()


# In[101]:


sns.countplot(x='Age',hue='Outcome',data=df)


# In[102]:


sns.countplot(x='Insulin',hue='Outcome',data=df)


# In[10]:


sns.countplot(x='Outcome',hue='Age',data=df)


# In[103]:


df.corr()


# In[104]:


sns.pairplot(df)


# In[105]:


sns.heatmap(df.corr())


# In[106]:


df['Pregnancies'].value_counts()


# In[19]:


import pandas_profiling


# In[107]:


df.describe()


# In[108]:


df['Glucose'].value_counts()


# In[109]:


df['BloodPressure'].value_counts()


# In[110]:


df['SkinThickness'].value_counts()


# In[126]:


df['Glucose'].value_counts()


# In[129]:


df['Insulin'].value_counts()


# In[134]:


df['Pregnancies'].value_counts()


# In[135]:


df['BMI'].value_counts()


# In[136]:


df['DiabetesPedigreeFunction'].value_counts()


# In[ ]:


def imputeage(cols):
    Age = cols[0]
        if pd.isnull(Age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 27
    else:
        return Age


# In[111]:


df.shape


# In[112]:


df.columns


# In[31]:


#df[df['Age']>21 ['Age']<=30]
#df[df['Age']<=30 & df['Age']>21]
#df['Agebin'] = 
pd.cut(df['Age'], [20, 30, 45,60,>60], labels=['20-30', '31-45', '46-60','60>'])


# In[45]:


for i in df['Age']:
    print([(df['Age']>=20) & (df['Age']<=30)])
    #print(df[(df['Age']>=20) & (df['Age']<=30)].Age)--Gives the AGe btw 21 and 30
    #print(df[(df['Age']>=20) & (df['Age']<=30)])--Gives the df of age btw 21 and 30
   # print(df['Age'])


# In[48]:


#df['new column name'] = df['column name'].apply(lambda x: 'value if condition is met' if x condition else 'value if condition is not met')
df['AgeBin']=df['Age'].apply(lambda x:  if ([(df['Age']>=20) & (df['Age']<=30)]) '20-30' else '>31')


# In[62]:


def imputeage(cols):
    Age = cols[0]
    Agebin = []
    if [(df['Age']>=20) & (df['Age']<=30)]:
        return Agebin.append(2030)#df['Agebin']='20-30'
    elif [(df['Age']>30) & (df['Age']<=45)]:
        return Agebin.append(3145)#df['Agebin']='31-45'
    elif [(df['Age']>45) & (df['Age']<=60)]:
        return Agebin.append(4660)#df['Agebin']='46-60'
    else:
        return Agebin.append(6100)#df['Agebin']='>61'
    


# In[137]:


#df.drop('binAge',axis=1,inplace=True)
df.columns


# In[120]:


#df.drop('BPbin',axis=1,inplace=True)
#df.drop('Skbin',axis=1,inplace=True)


# In[143]:


bins = [-1, 60, 80, 100, 200]
labels = ['0-60','61-80','81-100','>100']
df['BPbin'] = pd.cut(df['BloodPressure'], bins=bins, labels=labels)
print (df)


# In[123]:


bins = [-1, 10, 20, 40, 50,100]
labels = ['0-10','11-20','21-40','40-50','50-100']
df['Skbin'] = pd.cut(df['SkinThickness'], bins=bins, labels=labels)
print (df)


# In[140]:


#df['Agebin']= df['Age'].apply(imputeage,axis=1)
#df['Age'].apply(imputeage,)
bins = [20, 30, 45, 60, 200]
labels = ['20-30','31-45','46-60','>60']
df['binAge'] = pd.cut(df['Age'], bins=bins, labels=labels)
print (df)


# In[127]:



bins = [40, 75, 100, 130, 200]
labels = ['40-75','75-100','100-130','130-200']
df['binGlu'] = pd.cut(df['Glucose'], bins=bins, labels=labels)
print (df)


# In[130]:



bins = [-1, 100, 200, 300, 10000]
labels = ['0-100','100-200','200-300','>300']
df['binInsulin'] = pd.cut(df['Insulin'], bins=bins, labels=labels)
print (df)


# In[142]:


df.columns


# In[138]:


df.head(20)


# In[144]:


x=df.drop(['Outcome','Glucose','Insulin','SkinThickness','Age','BloodPressure'],axis=1)
y=df['Outcome']


# In[145]:


x.columns


# In[146]:


x.shape


# In[147]:


y.shape


# In[148]:


from sklearn.model_selection import train_test_split


# In[150]:


encodedDF = pd.get_dummies(df[['binAge','BPbin','Skbin','binGlu','binInsulin']])


# In[152]:


X = pd.concat([encodedDF,x.drop(['binAge', 'BPbin','Skbin', 'binGlu', 'binInsulin'],axis=1)],axis=1)


# In[151]:


encodedDF


# In[154]:


X.shape


# In[156]:


X.columns


# In[157]:


from sklearn.preprocessing import StandardScaler


# In[158]:


scx=StandardScaler()


# In[159]:


from sklearn.model_selection import train_test_split


# In[160]:


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=101)


# In[161]:


X_train.shape


# In[162]:


X_test.shape


# In[164]:


X_train_std = scx.fit_transform(X_train)


# In[165]:


X_train_std


# In[166]:


from sklearn.linear_model import LogisticRegression


# In[167]:


log = LogisticRegression()


# In[168]:


log.fit(X_train_std,y_train)


# In[169]:


log.intercept_


# In[170]:


log.coef_


# In[171]:


predict = log.predict(X_test)


# In[172]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score


# In[173]:


print(accuracy_score(y_test,predict))


# In[174]:


print(confusion_matrix(y_test,predict))


# In[175]:


print(classification_report(y_test,predict))


# In[176]:


print(roc_auc_score(y_test,predict))


# In[178]:


log.fit(X_train,y_train)


# In[179]:


log.intercept_


# In[180]:


log.coef_


# In[91]:


#x=df.drop(['Age'],axis=1,inplace=True)


# In[181]:


pred = log.predict(X_test)


# In[182]:


print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))


# In[183]:


print(classification_report(y_test,pred))


# In[184]:


print(roc_auc_score(y_test,pred))


# In[153]:


#X = pd.concat([encodedDF,x.drop(['Age'],axis=1)],axis=1)
#X = pd.concat([encodedDF,train.drop(['PassengerId'],axis=1)],axis=1)

