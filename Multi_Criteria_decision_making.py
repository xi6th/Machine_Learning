# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.arrays import categorical
import seaborn as sns

plt.rcParams['figure.figsize'] = (8,6)
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')

#Fetching data
""" 
    Download data from kaggle using this link
"""

loan = pd.read_csv("Train_Loan_Home.csv")
# print(loan.head())

# Load test data
loan_test = pd.read_csv("Test_Loan_Home.csv")

"""
    Understanding Data
"""

data_shape = loan.shape,loan_test.shape
data_info = loan.info(),loan_test.info()
data_types = loan.dtypes,loan_test.dtypes
missing_data = loan.isnull().sum(), loan_test.isnull().sum()

"""Heatmap to identify the features having null values"""
# sns.heatmap(loan.isnull(), yticklabels=False)
# plt.show()


# %%

"""
    Exploratory Data Analysis
"""

#dropping the unnecessary column
loan_mcdm_id = loan['Loan_ID']
loan.drop('Loan_ID', axis=1, inplace=True)
loan_test_id = loan_test['Loan_ID']
loan_mcdm_id = loan_test['Loan_ID']
loan_test.drop('Loan_ID', axis=1, inplace=True)

X = loan.drop('Loan_Status', axis=1)
y = loan['Loan_Status']


"""
    Categorizing datatypes
"""
categorical_features = []
numerical_features = []
for i in X.columns.tolist():
    if X[i].dtype=='object':
        categorical_features.append(i)
    else:
        numerical_features.append(i)

# print(categorical_features)
# print(numerical_features)

""" 
    Univariate Analysis
"""
loan_counts = loan["Loan_Status"].value_counts()
# print(loan_counts)

loan['Loan_Status'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Loan Status')
plt.show()

# %%
for i in categorical_features:
    print('Features: ', i)
    print(X[i].value_counts(normalize=True))
    X[i].value_counts(normalize=True).plot(kind='bar')
    plt.xlabel(i)
    plt.savefig(i,dpi=300, bbox_inches='tight')
    plt.show()
    print('\n')
    
"""
    From The above visualizations it can be inferred that:

Around 81% customers are Male.
65% customers are Married.
Most of the customers don't have any dependents.
78% customers are Graduate.
Only 14% customers are self employed.
38% customers are from semiurban area, 33% are from urban area, 29% are from rural area
"""

#Numerical Features
sns.distplot(X['ApplicantIncome'], bins=50, kde=True)
plt.savefig('ApplicantIncome2.png', dpi=300, bbox_inches='tight')
plt.show()
X['ApplicantIncome'].plot(kind='box')
plt.savefig('ApplicantIncome.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# From the above visualization it can be inferred that:
# 
# 1. The Applicant Income Feature does not follow normal distribution, Most of the Income ranges from 0-20,000
# 
# 2. The boxplot surely indicates that the feature is having many Outliers. It is due to different income labels of different customers. We can group the income of the customers with their education label

# %%
X.boxplot(column='ApplicantIncome',by='Education')
plt.show()

# %% [markdown]
# Most of the Customers who are graduate is having very high incomes

# %%
sns.distplot(X['CoapplicantIncome'],kde=True)
plt.savefig('CoapplicantIncome_dist.png', dpi=300, bbox_inches='tight')
plt.show()
X.boxplot(column='CoapplicantIncome')
plt.savefig('CoapplicantIncome_box.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# 1. This feature is having a Right Skewed Distribution and most of the CoapplicantIncome ranges from 0-6000
# 2. Also the feature is having few outliers

# %%
sns.distplot(X['LoanAmount'], kde=True)
plt.savefig('LoanAmount_dist.png', dpi=300, bbox_inches='tight')
plt.show()
X['LoanAmount'].plot(kind='box')
plt.savefig('LoanAmount_box.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# LoanAmount feature follows Normal Distribution but it is having many Outliers

# %% [markdown]
# The features Loan_Amount_Term and Credit_History are having categorical values, so we will consider them in the categorical features list

# %%
X['Loan_Amount_Term'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Loan Amount Term')
plt.savefig('Loan_Amount_Term.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Around 85% loans are having 360 as their loan amount term

# %% [markdown]
# Bivariate Analysis

# %%
gender = pd.crosstab(loan['Gender'], loan['Loan_Status'])
gender.div(gender.sum(axis=1), axis=0).plot(kind='bar', stacked=True)
plt.savefig('gender_bivarent.png', dpi=300, bbox_inches='tight')
plt.show()

married = pd.crosstab(loan['Married'], loan['Loan_Status'])
married.div(married.sum(axis=1), axis=0).plot(kind='bar', stacked=True)
plt.savefig('married_bivarent.png')

dependents=pd.crosstab(loan['Dependents'],loan['Loan_Status'])
dependents.div(dependents.sum(axis=1),axis=0).plot(kind='bar',stacked=True)
plt.show()
education=pd.crosstab(loan['Education'],loan['Loan_Status'])
education.div(education.sum(axis=1),axis=0).plot(kind='bar',stacked=True)
plt.show()
self_employed=pd.crosstab(loan['Self_Employed'],loan['Loan_Status'])
self_employed.div(self_employed.sum(axis=1),axis=0).plot(kind='bar',stacked=True)
plt.show()
property_area=pd.crosstab(loan['Property_Area'],loan['Loan_Status'])
property_area.div(property_area.sum(axis=1),axis=0).plot(kind='bar',stacked=True)
plt.show()
credit_history=pd.crosstab(loan['Credit_History'],loan['Loan_Status'])
credit_history.div(credit_history.sum(axis=1),axis=0).plot(kind='bar',stacked=True)
plt.show()

# %% [markdown]
# From the above visualization this following points can be inferred:
# 
# 1. The percentage of female and male customers,getting loan approval are same.
# 2. Married customers are more likely to get the loan approval.
# 3. Customers who have 1 and 3+ dependents are having more chance to get the loan approval.
# 4. Graduate customers are more likely to get the loan approval.
# 5. The percentage of self employed and not employed customers,getting loan approval are same.
# 6. Customers from semi urban area are having more chance to get loan approval.
# 7. Customers with credit score 1 are more likely to get loan approval.

# %% [markdown]
# Handling Missing Values

# %%
print("Missing Vales:" ,'\n',loan.isnull().sum())

# %% [markdown]
# To handle the null values:
# 1. For the Categorical Features the null values will be replaced by the mode value
# 2. For the Numerical features the null values will be replaced by the mean or by the media value

# %%
X["Gender"].replace(np.nan, X['Gender'].mode()[0], inplace=True)
X['Married'].replace(np.nan,X['Married'].mode()[0], inplace=True)
X['Dependents'].replace(np.nan,X['Dependents'].mode()[0], inplace=True)
X['Self_Employed'].replace(np.nan,X['Self_Employed'].mode()[0],inplace=True)
X['Loan_Amount_Term'].replace(np.nan,X['Loan_Amount_Term'].mode()[0], inplace=True)
X['Credit_History'].replace(np.nan,X['Credit_History'].mode()[0], inplace=True)


# %% [markdown]
# As the Loan Amount feature has many outliers, we will replace the null values with median

# %%
X['LoanAmount'].replace(np.nan, X['LoanAmount'].median(), inplace=True)

# %% [markdown]
# Verify if we still have missing values in the dataset

# %%
X.isnull().sum()

# %% [markdown]
# All the null values has been removed

# %% [markdown]
# Outlier Treatment
# As the Loan Amount feature follows right skewed distribution, we will perform a log transformation to get the normal distribution, as the model will a better performance on the normal distribution. The same log transformation will be applied on the test data

# %%
X['LoanAmount_Log']=np.log(X['LoanAmount'])
sns.distplot(X['LoanAmount_Log'], bins=50)
plt.savefig('LoanAmount_Log.png', dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# The log transformation of the Loan Amount feature follows normal distribution

# %%
#dropping the Loan Amount Feature
X.drop('LoanAmount', axis=1, inplace=True)

# %% [markdown]
# Feature Engineering 
# Convecting all the categorical variables into numerical variables

# %%
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Property_Area','Loan_Amount_Term','Credit_History']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in categorical_features:
    X[i] = le.fit_transform(X[i])

# %%
X.head()

# %% [markdown]
# Feature Selection
# 
# Correlation Matrix

# %%
plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.savefig('Corrolation Map.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Feature Importance

# %%
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(X,y)
values=pd.Series(etc.feature_importances_)
c = X.columns
w = values

# %%
values.index=X.columns
values.sort_values(ascending=False).plot(kind='barh')
plt.savefig('Feature Importances.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Data Normalization

# %%
from sklearn.preprocessing import StandardScaler
X_norm = StandardScaler().fit_transform(X)
X_norm[0:5]

# %% [markdown]
# Model Creation
# The following steps to be followed:
#     1. Apply all the changes in test dataset
#     2. Separate the training dataset into train data and validation data
#     3. Apply different Machine Learning Classification Algorithm to train the dataset
#     4. Check the performance matrix based on the output of the Validation dataset
#     5. Apply the best Model into the test data

# %% [markdown]
# Applying all the changes in the Test dataset

# %%
loan_test.head()

# %% [markdown]
# ApplicantIncomeMonthly,CoapplicantIncomeMonthly,LoanAmountThousands,Loan_Amount_Term_Months these four features are not having the same name in the training data. So renaming the features with the same name in the training dataset

# %%
loan_test.rename(columns={'ApplicantIncomeMonthly':'ApplicantIncome','CoapplicantIncomeMonthly':'CoapplicantIncome','LoanAmountThousands':'LoanAmount','Loan_Amount_Term_Months':'Loan_Amount_Term'},inplace=True)

# %%
loan_test.isnull().sum()

# %%
loan_test['Gender'].replace(np.nan,loan_test['Gender'].mode()[0],inplace=True)
loan_test['Dependents'].replace(np.nan,loan_test['Dependents'].mode()[0],inplace=True)
loan_test['Self_Employed'].replace(np.nan,loan_test['Self_Employed'].mode()[0],inplace=True)
loan_test['LoanAmount'].replace(np.nan,loan_test['LoanAmount'].median(),inplace=True)
loan_test['Loan_Amount_Term'].replace(np.nan,loan_test['Loan_Amount_Term'].mode()[0],inplace=True)
loan_test['Credit_History'].replace(np.nan,loan_test['Credit_History'].mode()[0],inplace=True)

# %%
loan_test.isnull().sum()

# %%
for i in categorical_features:
    loan_test[i]=LabelEncoder().fit_transform(loan_test[i])
loan_test.head()

# %%
loan_test['LoanAmount_log']=np.log(loan_test['LoanAmount'])
loan_test.drop('LoanAmount',axis=1,inplace=True)
loan_test.head()

# %%
loan_test_norm=StandardScaler().fit_transform(loan_test)
loan_test_norm[0:5]

# %% [markdown]
# Training and Validation Data Split

# %%
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(X_norm,y,test_size=0.25, random_state=42)
x_train.shape,y_train.shape,x_val.shape,y_val.shape


# %% [markdown]
# Model Creation
# The following models will the applied on the training data:
# 1. Logistic regression
# 2. Support Vector Classifier
# 3. Decision Tree Classifier
# 4. Random Forest Classifier
# 5. Xgboost Classifier

# %%
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score

# %%
accuracy=[]
f1=[]
model=[]

# %%
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_hat=lr.predict(x_val)

# %%
accuracy.append(np.round(accuracy_score(y_val,y_hat),2))
f1.append(np.round(f1_score(y_val,y_hat,average='weighted'),2))
model.append('Logistic Regression')

# %%
sns.heatmap(confusion_matrix(y_val,y_hat), annot=True, fmt='.0f')
plt.show()

# %% [markdown]
# Support Vector Machine

# %%
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_hat=svc.predict(x_val)

# %%
accuracy.append(np.round(accuracy_score(y_val,y_hat),2))
f1.append(np.round(f1_score(y_val,y_hat,average='weighted'),2))
model.append('SVC')

# %%
sns.heatmap(confusion_matrix(y_val,y_hat),annot=True, fmt='.0f')
plt.show()

# %% [markdown]
# Decision Tree Classifier

# %%
from sklearn.tree import DecisionTreeClassifier
dst = DecisionTreeClassifier(criterion='entropy')
dst.fit(x_train,y_train)
y_hat = dst.predict(x_val)

# %%
accuracy.append(np.round(accuracy_score(y_val,y_hat),2))
f1.append(np.round(f1_score(y_val,y_hat,average='weighted'),2))
model.append('Decision Tree')

# %%
sns.heatmap(confusion_matrix(y_val,y_hat), annot=True, fmt='.0f')
plt.show()

# %% [markdown]
# Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_hat=rfc.predict(x_val)

# %%
accuracy.append(np.round(accuracy_score(y_val,y_hat),2))
f1.append(np.round(f1_score(y_val,y_hat,average='weighted'),2))
model.append('Random Forest')

# %%
sns.heatmap(confusion_matrix(y_val,y_hat), annot=True, fmt='.0f')
plt.show()

# %% [markdown]
# XGBClassifer

# %%
from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=100,max_depth=3)
xgb.fit(x_train,y_train)
y_hat=xgb.predict(x_val)

# %%
accuracy.append(np.round(accuracy_score(y_val,y_hat),2))
f1.append(np.round(f1_score(y_val,y_hat,average='weighted'),2))
model.append('Xgboost')

# %%
sns.heatmap(confusion_matrix(y_val,y_hat), annot=True, fmt='.0f')
plt.show()

# %%
model

# %%
output = pd.DataFrame({
    'Model': model,
    'Accuracy':accuracy,
    'F1 score': f1
})

# %%
output

# %%
y_pred = svc.predict(loan_test_norm)
y_pred[0:5]

# %%
result = pd.DataFrame({
    'LoanID': loan_test_id,
    'Loan_Status':y_pred
})
result

file_name = 'LoanPrediction.xlsx'
result.to_excel(file_name)

# files = pd.DataFrame(eval(rank))


# %% [markdown]
# Multi Criteria Decision

# %%
m = loan_test_norm
m

# %% [markdown]
# Convert Arrays to matrix

# %%
matrix = np.asmatrix(m)
matrix

# %%
w

# %%
objectives = [min,min,min,min,min,max,max,min,max,min,max]
print(objectives)

# %%
import skcriteria  as skc

dm = skc.mkdm(matrix,[max,max,min,max,min,max,min,max,max,max,min])
dm

# %%
dm = skc.mkdm(
    matrix,
    objectives,
    weights=w,
    alternatives = loan_mcdm_id,
    criteria = c
)
dm

# %%
dm

# %%
dm.weights

# %%
dmt = dm
dmt

# %%
# scaler = scalers.SumScaler(target='both')
# dmt = scaler.transform(dm)
# dmt

# %%
import matplotlib.pyplot as plt
# we create 2 axis with the same y axis
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# in the first axis we plot the criteria KDE
dmt.plot.kde(ax=axs[0])
axs[0].set_title("Criteria")
dmt.plot.wbar(ax=axs[1])
axs[1].set_title("Weights")
# adjust the layout of the figute based on the content
fig.tight_layout()
plt.show()

# %%
from skcriteria.madm import similarity # here lives TOPSIS
from skcriteria.pipeline import mkpipe 
from skcriteria.preprocessing import invert_objectives, scalers

pipe = mkpipe(invert_objectives.MinimizeToMaximize(),
scalers.VectorScaler(target="matrix"), # this scaler transform the matrix
scalers.SumScaler(target="weights"), # and this transform the weights
similarity.TOPSIS(),)
pipe

# %%
rank = pipe.evaluate(dm)
# df = pd.DataFrame(list(rank.values()), index=rank.keys())
# df
rank

# %%


# %%



