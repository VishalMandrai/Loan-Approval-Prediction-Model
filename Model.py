import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import random
from collections import Counter
import pickle

df = pd.read_csv('data_preprocessed')
df.drop('Unnamed: 0' , axis = 1 , inplace = True)
df.head()


# ## --------------------------------------------------------------

# # FEATURE ENGINEERING

# ## Annual Income:

## Creating a new feature with all NAN values replaced.....
## Applying Mean Value Imputation Technique...

df['Annual_Income_Changed'] = df['Annual_Income']
df['Annual_Income_Changed'] = df['Annual_Income_Changed'].fillna(df['Annual_Income'].mean())

## Since there are two many Outliers above 200000, which are affecting the model making process hence we should remove them...
df = df[df['Annual_Income'] < 200000]

## Reseting the index.....
df.reset_index(drop = True  , inplace = True)


# ## Current Loan Amount:
# - Here, In "Current.Loan.Amount" feature we have 7397 NAN entries. This NAN are represented by 
#   "99999999". We need to correct this problem now.

## Creating a Sample Series of all the entries other than '99999999'....
tempo = df[df['Current_Loan_Amount'] != 99999999]
random_entries = tempo[tempo['Loan_Status'] == 1].Current_Loan_Amount.sample(df['Current_Loan_Amount'].value_counts().max()
                                                                             , random_state=9)
random_entries.index = df[df['Current_Loan_Amount'] == 99999999].index

## Creating an array of bool dtype, 99999999 = True | Elements other than 99999999 = False
m = np.where(df['Current_Loan_Amount'].values == 99999999 , True , False )

## Changing all '99999999' with random values selected from the feature column.
df['Current_Loan_Amount_Changed'] = df['Current_Loan_Amount']
df.loc[m , 'Current_Loan_Amount_Changed'] = random_entries

# - Now all NAN are replaced with Random Entries from the feature column. Also we got the similar variance distribution too.


# ## Credit Score:

## Now we will apply RANDOM SAMPLE IMPUTATION to rectify NAN...
temp1 = df['Credit_Score'].dropna().sample(df['Credit_Score'].isnull().sum())
temp1.index = df[df['Credit_Score'].isnull()].index

## 'Credit_Score_Changed' is the feature with NAN replaced...
df['Credit_Score_Changed'] = df['Credit_Score']
df.loc[df['Credit_Score'].isnull() , 'Credit_Score_Changed'] = temp1 


## Since there are two many Outliers above 650, which are affecting the model making process hence we should remove them...
df = df[df['Credit_Score_Changed'] > 650]

## Reseting the index.....
df.reset_index(drop = True  , inplace = True)

# ## Purpose:
# 

#     - Combining some categories like "Business Loan" , "Buy House" , "Medical Bills", "major_purchase" and "small_business"  because these categories have some correlation with respect to the fact that the Loans are taken for High Capital Investment.
#     - There are some Categories which can also be combined togther like : "Other" , "Take a Trip" , "moving" ,  "Educational Expenses" , "vacation" , "wedding" , "renewable_energy". All these category are very arbitrary and don't have any serious trend, so we can have one sepearte category for this.
# 
# #### **Now we have 4 categories:**
# 
# - Category 1: Debt Consolidation
# - Category 2: High Capital = "Business Loan" + "Buy House" + "Medical Bills" + "major_purchase" + "small_business" 
# - Category 3: Arbitrary Purpose = "Other" + "Take a Trip" + "moving" + "Educational Expenses" + "vacation" + "wedding" + "renewable_energy" + 'Buy a Car'
# - Category 4: Home Improvements


## Creating a sepearte Feature with above Changes....
df['Purpose_Changed'] = df['Purpose']

df['Purpose_Changed'].replace(("Business Loan" , "Buy House" , "Medical Bills" , "major_purchase" , "small_business")
                              , 'High Capital' , inplace = True)

df['Purpose_Changed'].replace(("Other" , "Take a Trip" , "moving" , "Educational Expenses" , "vacation" , "wedding" , 
                               "renewable_energy" , 'Buy a Car') , 'Arbitrary Purpose' , inplace = True)


temp = pd.DataFrame({'Count' : df['Purpose_Changed'].value_counts() , 
                     'Percentage of Loan Approved' : df.groupby(['Purpose_Changed']).Loan_Status.mean()}
                    , index = df['Purpose_Changed'].value_counts().index)


## We can use Probability Imputation technique.....

cate_count = dict(df['Purpose_Changed'].value_counts()/df.shape[0])
cate = list(cate_count.keys())
values = list(cate_count.values())

for i in range(len(cate_count.keys())):
    df['Purpose_Changed'].replace(cate[i]  , values[i] , inplace = True)


# ## Home Ownership :

## Creating dummy variables to represent "Home_Ownership" Categories....
Home_Ownership_dummy = pd.get_dummies(df['Home_Ownership'] , drop_first = True , prefix='Home_Ownership', prefix_sep='/')
df = pd.concat([df,Home_Ownership_dummy] , axis = 1)


# ## Term:

## Creating dummy variables to represent "Term" Categories....
Term_dummy = pd.get_dummies(df['Term'] , drop_first = True , prefix='Term', prefix_sep='->')
df = pd.concat([df,Term_dummy] , axis = 1)


# ## Years in Current Job :

## Selecting top 4 most occuring elements in the feature....
sample_temp = list(df['Years_in_current_job'].value_counts().head(4).index)

## Creating the sample for imputation in NAN places....
sample = random.choices(sample_temp, weights = [2, 1, 1, 1], k = df['Years_in_current_job'].isnull().sum())
## Converting List to Series....
sample = pd.Series(sample)

## Matching index of all NAN position with the "sample" Series
sample.index = df[df['Years_in_current_job'].isnull()].index

## Imputing "sample" data in the main feature....
df.loc[df['Years_in_current_job'].isnull() , 'Years_in_current_job'] = sample


## pairing different entries.....
cate_1 = ['less than  1 year' , '1 year' , '2 years', '3 years']
cate_2 = ['4 years', '5 years' , '6 years' , '7 years']
cate_3 = ['8 years', '9 years' , '10+ years']


for i in cate_1:
    df['Years_in_current_job'].replace(i , 'cate_1' , inplace = True)
for i in cate_2:
    df['Years_in_current_job'].replace(i , 'cate_2' , inplace = True)
for i in cate_3:
    df['Years_in_current_job'].replace(i , 'cate_3' , inplace = True)


## Creating dummy variables to represent "Years_in_current_job" Categories....
Years_in_current_job_dummy = pd.get_dummies(df['Years_in_current_job'] , drop_first = True , 
                                            prefix='Years_in_current_job', prefix_sep='->')
df = pd.concat([df,Years_in_current_job_dummy] , axis = 1)


# ## Months Since Last Delinquent :

## Due to very low Corr Coef. and very Large number of NAN values in 'Months_since_last_delinquent'.....
## We should better drop the feature....
df = df.drop(['Months_since_last_delinquent'] , axis = 1)


# ## Bankruptcies :

## since, "0" is the most highly occuring value in the feature "Bankruptcies"....
## we can better fill all NAN with "0" value....
df['Bankruptcies'].fillna("0" , inplace = True)


# ## Tax Liens :

## since, "0" is the most highly occuring value in the feature "Tax_Liens"....
## we can better fill all NAN with "0" value....
df['Tax_Liens'].fillna("0" , inplace = True)


# ## ------------------------------------------------------------------------

# ### Dropping all unwanted columns :

df.drop(['Loan_ID', 'Current_Loan_Amount', 'Term', 'Credit_Score', 'Years_in_current_job', 'Home_Ownership', 
 'Annual_Income', 'Purpose' , 'Purpose_Changed'] , axis = 1 , inplace = True)

## NOW are dat frame is ready and now we can prepare the ML Model.....


# ## ------------------------------------------------------------------------

# ### Feature Selection:

from sklearn.feature_selection import f_regression

## Calculating P Values for each Features....
f_val , p_val = f_regression(df.drop('Loan_Status' , axis = 1) , df['Loan_Status'])

## Creating the list for Features with P values less more than 0.05....
## These features can be drop here....
drop = []

for i in range(len(p_val)):
    ## Filtering features with respect to P Values....
    if p_val[i] > 0.05:
        drop.append(df.drop('Loan_Status' , axis = 1).columns.values[i])
    else:
        pass

## Droppinf all selected features in "drop"....
df.drop(drop , axis = 1 , inplace = True)

# ## ------------------------------------------------------------------------

# ## Feature Scaling:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('Loan_Status' , axis = 1))

input_scaled = scaler.transform(df.drop('Loan_Status' , axis = 1))

# Saving Scaler to disk.....
pickle.dump(scaler, open('scaler.pkl','wb'))


## Creating dataframe of all scaled feartures....
input_df = pd.DataFrame(data = input_scaled , columns = df.drop('Loan_Status' , axis = 1).columns.values)


# ## ------------------------------------------------------------------------

# #### Apply Model Selection:

from sklearn.model_selection import train_test_split
## Splitting the dataset in two halves:
## 1. Train Set
## 2. Test Set
x_train , x_test , y_train , y_test  = train_test_split(input_df , df['Loan_Status'] , random_state = 55 , test_size = 0.25)


# ## -----------------------------------------------------------------------

# ## Model:

# #### Using Random Forest as Classifier :

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_jobs = -1 , random_state= 45)

rf_model.fit(x_train , y_train)

# ## -----------------------------------------------------------------------

# #### Using Gradient Boosting as Classifier :

from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state= 45)

gb_model.fit(x_train , y_train)

# ## -----------------------------------------------------------------------
## Pickeling File....
# Saving RANDOM FOREST MODEL to disk
pickle.dump(rf_model, open('model_rf.pkl','wb'))

# Loading RANDOM FOREST MODEL to compare the results
rf_model = pickle.load(open('model_rf.pkl','rb'))

# ## ----------------------------

# Saving GRADIENT BOOSTING MODEL to disk
pickle.dump(gb_model, open('model_gb.pkl','wb'))

# Loading GRADIENT BOOSTING MODEL to compare the results
gb_model = pickle.load(open('model_gb.pkl','rb'))









