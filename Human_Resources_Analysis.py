"""
    The Analysis is divided into two majors:
        1. People Analytics with Attrition Predictions
        2. Multi criteria decision making
        
        People Analytics with Attrition Predictions:
            Every year a lot of companies hire a number of employees. The companies invest time and money in training those employees, 
            not just this but there are training programs within the companies for their existing employees as well. 
            The aim of these programs is to increase the effectiveness of their employees. 
            But where HR Analytics fit in this? and is it just about improving the performance of employees?
            
            HR Analytics
                Human resource analytics (HR analytics) is an area in the field of analytics that refers to applying analytic 
                processes to the human resource department of an organization in the hope of improving employee performance and t
                herefore getting a better return on investment. HR analytics does not just deal with gathering data on employee efficiency. 
                Instead, it aims to provide insight into each process by gathering data and then using it to make relevant decisions 
                about how to improve these processes.
                
                Attrition in HR
                Attrition in human resources refers to the gradual loss of employees over time. In general, 
                relatively high attrition is problematic for companies. HR professionals often assume a 
                leadership role in designing company compensation programs, work culture and motivation systems that help 
                the organization retain top employees.
                
                How does Attrition affect companies? and how does HR Analytics help in analyzing attrition? 
                We will discuss the first question here and for the second question we will write the code and try
                to understand the process step by step.
                
                Attrition affecting Companies
                A major problem in high employee attrition is its cost to an organization.
                Job postings, hiring processes, paperwork and new hire training are some of the common 
                expenses of losing employees and replacing them. Additionally, regular employee turnover p
                rohibits your organization from increasing its collective knowledge base and experience over time.
                This is especially concerning if your business is customer facing, as customers often prefer to interact 
                with familiar people. Errors and issues are more likely if you constantly have new workers.
                Hope the basics made sense. Let’s move on to coding and try finding out how HR Analytics help in understanding attrition.
"""

# importing required python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



#Finance Company Loan data

""" 
    fetching data
    download data from kaggle 
"""

hr_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
read_first_five_data = (hr_data.head())

staff_names = pd.read_csv("Staff_names.csv")
company_employees =  pd.concat([hr_data, staff_names], axis=1, join="inner")
# print(company_employees)

# data check for missing values
missing = hr_data.isnull().sum()
# print(hr_data.columns)
# print(missing)

# Checking data structure
structure = hr_data.dtypes
# print(structure)

"""
    Categorizing datatypes
"""
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
hr_data_num = hr_data.select_dtypes(include=numerics)
# print(hr_data_num.head())

"""
    Data Visualization:
    
        A diverging color palette that has markedly different colors at the two ends of the value-range with a pale, 
        almost colorless midpoint, works much better with correlation heatmaps than the default colormap.
        While illustrating this statement, let’s add one more little detail: how to save a heatmap to 
        a png file with all the x- and y- labels (xticklabels and yticklabels) visible.
        
        save heatmap as .png file
        dpi - sets the resolution of the saved image in dots/inches
        bbox_inches - when set to 'tight' - does not allow the labels to be cropped
"""
# corrola = hr_data.corr()
# plt.figure(figsize=(16, 6))
# heatmap = sns.heatmap(corrola, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

# plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()

"""
    Feature Engineering:
    
        Let’s have a look at the data and see how features are contributing in the data and in attrition of employees. 
        We need to first check the data type of the features, 
        why? Because we can only see the distribution of numerical/continuous values in a dataset. 
        In order to take a peak into categorical/object values we have to bind them with a numeric 
        variable and then you will be able to see their 
        relevance to the dataset or you can replace the categorical variable with dummies.
        
        From the above heat map we can now see which variables are poorly 
        correlated and which ones are strongly correlated.
"""

#Lets extract the strongly correlated variables
hr_data_uc = hr_data[['Age','DailyRate','DistanceFromHome', 
                    'EnvironmentSatisfaction', 'HourlyRate',                     
                    'JobInvolvement', 'JobLevel',
                    'JobSatisfaction', 
                    'RelationshipSatisfaction', 
                    'StockOptionLevel',
                    'TrainingTimesLastYear']].copy()
# print(hr_data_uc.head())

"""
        Stop for a sec and think what we don’t have it here- two things, One, 
        Categorical variable and any information about attrition. Let’s combined 
        those with above dataframe.
"""

#Copy categorical data
hr_data_cat = hr_data[['Attrition', 'BusinessTravel','Department',
                    'EducationField','Gender','JobRole',
                    'MaritalStatus',
                    'Over18', 'OverTime']].copy()
# print(hr_data_cat.head())

"""
    Replace Yes and No in Attrition with 1 and 0.
"""
num_val = {'Yes':1, 'No':0}
hr_data_cat['Attrition'] = hr_data_cat["Attrition"].apply(lambda x: num_val[x])
# print(hr_data_cat.head())

"""
    Now replace other categorical variables with dummy values.
"""

hr_data_cat = pd.get_dummies(hr_data_cat)
# print(hr_data_cat.head())

"""
    Now that we have all the data in numerical format, 
    we can now combine hr_data_num and hr_data_cat.
"""
hr_cleaned_data = pd.concat([hr_data_num, hr_data_cat], axis=1)
# print(hr_cleaned_data["EmployeeNumber"])

scaler = preprocessing.MinMaxScaler()
names = hr_cleaned_data.columns
d = scaler.fit_transform(hr_cleaned_data)
hr_data_final = pd.DataFrame(d, columns=names)
# print(scaled_df["EmployeeNumber"])

"""
    Modelling the data
        We have our final dataset. We now have to start modelling- Predicting the Attrition.
        Wait? Are you also confused like me? We already have the Attrition data then what is it here to predict? 
        Well most of the time in Regression and classification problem, you run your model with the available values 
        and check the metrics like accuracy of the model by comparing observed values with true values.
        If you won’t have the true values how would you know that the predictions are correct.
        
        Now you will realize that, how important the training data phase is.
        We train the model in a way that it can predict(almost) correct results.
        In this dataset, we don’t have any missing values for Attrition, 
        we will split the data into train and test. We will train the model 
        on training data and predict the results on test data.
        
        For this particular exercise we will use Random Forest Classifier. 
        Before jumping into code, let’s get a little background about the RF classifier.
    
    Random Forest Classifier
        Number of weak estimators when combined forms strong estimator.
        The Random forest works on Bagging principle. It is an ensemble of Decision Trees. 
        The bagging method is used to increases the overall results by combining weak models. 
        How does it combine the results? In the case of Classification problem it takes 
        the mode of the classes, predicted in the bagging process.
        I bet, a lot of you must be thinking, why did we choose the Random Forest
        for this particular problem? Well, to answer that, first let’s write the code.
"""

# print(scaled_df.shape)

target = hr_data_final['Attrition']
features = hr_data_final.drop('Attrition', axis=1)

#create the train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.4,
                                                    random_state=10)

#Create the model and train
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Predict the results for the test
test_pred = model.predict(X_test)

#test the accuracy
accuracy_score = accuracy_score(y_test, test_pred)
# print(accuracy_score)

# feat_importances = pd.Series(model.feature_importances_, index=features.columns)
# feat_importances = feat_importances.values
# feat_importances.plot(kind='barh')
# plt.show()

feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances = feat_importances.nsmallest(20)
feat_importances.plot(kind='barh')
plt.show()

#sum of each values
#print(sum(feat_importances))
"""
    converting dataframes to 2d matrix    
"""
# m = np.asmatrix(scaled_df)
# print(m)

