
# coding: utf-8

# In[881]:


import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from datetime import datetime
import math
import scipy.stats
import random
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


plt.style.use('ggplot')
random.seed(123)
np.set_printoptions(suppress=True)

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# In[708]:


# Loan_id - A unique loan number assigned to each loan customers

# Loan_status - Whether a loan is paid off, in collection, new customer yet to payoff, or paid off after the collection efforts

# Principal - Basic principal loan amount at the origination

# terms - Can be weekly (7 days), biweekly, and monthly payoff schedule

# Effective_date - When the loan got originated and took effects

# Due_date - Since it’s one-time payoff schedule, each loan has one single due date

# Paidoff_time - The actual time a customer pays off the loan

# Pastdue_days - How many days a loan has been past due

# Age, education, gender A customer’s basic demographic information


# In[709]:


# Hypothesis Testing tool kit

def draw_bs_reps(data, func, size = 1):
    """Draw bootstrap replicates"""
    
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = func(np.random.choice(data, len(data)))
    return bs_replicates

def draw_bs_ts(data1, data2, orig_data, size  = 1):
    data1_shifted = data1 - np.mean(data1) + np.mean(orig_data)
    data2_shifted = data2 - np.mean(data2) + np.mean(orig_data)
    
    bs_replicates1 = draw_bs_reps(data1, np.mean, 10000)
    bs_replicates2 = draw_bs_reps(data2, np.mean, 10000)
    
    bs_replicates_diff = bs_replicates1 - bs_replicates2
    empirical_diff_means_age = diff_of_means(data1, data2)

    p = np.sum(bs_replicates_diff >= empirical_diff_means_age)/len(bs_replicates_diff)
    print('Two sample test statistic p value = ',p)


def conf_int(data, crit = 5):
    return np.percentile(data, [0 + crit/2, 100 - crit/2])

def norm_test(data, alpha = .05):
    """Test data for normality: Null hypothesis data normally distributed"""
    
    if scipy.stats.mstats.normaltest(data).pvalue < alpha:
        print("Sample Not Normally Distributed")
    else:
        print("Sample Normally Distributed ")
    print("P Value: " + str(scipy.stats.mstats.normaltest(data).pvalue),
          "Mean: " + str(np.mean(data)),
          "Std: " + str(np.std(data)),
          "Conf Int: " + str(conf_int(data))         
         ) 


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays"""
    
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

def chi_test(data):
    ct = pd.crosstab(data, df.late)
    print('Chi Squared p value = ', scipy.stats.chi2_contingency(ct)[1])
    display(ct)


# In[710]:


print(df.isnull().sum())
df.groupby('loan_status')['loan_status'].count()


# In[711]:


# Import data and Read data

df = pd.read_csv('C:/Users/dmitr/Documents/Work/Projects/Kaggle/Loan Data/Loan payments data.csv')
df['education'] = df['education'].replace(to_replace = 'Bechalor', value = 'Bachelor')
df.rename(columns={'Principal': 'principal', 'Gender': 'gender'}, inplace=True)

display(df.info())
df.head()


# In[712]:


# Data cleaning 

object_cols = ['loan_status','education','gender']
date_cols = ['effective_date','due_date','paid_off_time']
df[object_cols] = df[object_cols].apply(lambda x: x.str.lower(), axis = 1)
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df['days_overdue'] = (df.paid_off_time - df.due_date).astype('timedelta64[D]').fillna(df.past_due_days)
df['late'] = df['days_overdue'].apply(lambda x: 0 if x <= 0 else 1)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)

df = df.drop(date_cols + ['past_due_days','Loan_ID'], axis = 1)
df.head()


# In[713]:


# Add levels to ordinal values

print(df.education.unique())

df.education = df.education.astype(pd.api.types.CategoricalDtype(
    ordered=True,categories=['high school or below', 'college', 'bachelor', 'master or above']))

df.education.cat


# In[714]:


# Let us observe two populations: 1. Late on repaying loan 2. Not late on repaying loan


df_late = df[df.late == 1]
df_notlate = df[df.late == 0]

display_side_by_side(
    df_late.describe().round(),
    df_notlate.describe().round()
)

display_side_by_side(
    df_late.select_dtypes(include='object').describe(),
    df_notlate.select_dtypes(include='object').describe()
)


# In[715]:


# Null Hypothesis: Late and Not Late groups both have the same mean age
# Test statistic: Mean 

# Let us begin with age. Descriptive analytics show us that age is not normally distributed in either sample, but looks identically distributed let us confirm
norm_test(df_late.age)
norm_test(df_notlate.age)

_ = sns.violinplot(x="late", y="age", data=df, inner=None)
_ = sns.swarmplot(x="late", y="age", data=df, color="w", alpha=.5)
_ = plt.xlabel('late status')
_ = plt.ylabel('age')
_ = plt.legend(['not late', 'late'])

plt.show()

# Kolmogorov-Smirnov statistic, based on large p value we cannot reject that both samples are not distributed identically 
print("Kolmogorov-Smirnov p value: ",scipy.stats.ks_2samp(df_late.age, df_notlate.age).pvalue)

# Using bootstrap method we reject the alternative hypothesis that the two groups' age means are significantly different
draw_bs_ts(df_late.age, df_notlate.age, df.age, 10000)


# In[716]:


# Null hypothesis: Categorical principal, education, gender, terms are identically distributed amongst late and non late groups

# Chi Square on ordinal values, contingency table of observed counts
# Assumption #1: Two variables measured at an ordinal or nominal level
# Assumption #2: Two variable should consist of two or more categorical, independent groups
# Assumption #3: Each category excedes count of 5

# Education is not statistically significant

print('Education')

education_grp = df.education.map(lambda x: 'bachelor or above' if x in ['bachelor','master or above'] else x)

display(pd.crosstab(df.education, df.late))
chi_test(education_grp)


# In[717]:


# terms is not statistically significant

print('terms')
display(pd.crosstab(df.terms, df.late))
terms_grp = df.terms.map(lambda x: '15 or less' if x <= 15 else '30')
chi_test(terms_grp)


# In[718]:


# principle is not statistically significant

print('principal')
display(pd.crosstab(df.principal, df.late))
principal_grp = df.principal.map(lambda x: 'less than 1000' if x < 1000 else '1000')
chi_test(principal_grp)


# In[719]:


# Fisher Test for 2 by 2 of observed counts 
# Assumption 1: 2 x 2 matrix
# Assumption 2: Each category > 5

# gender is not statistically significant

print('gender')
print('Fisher test p value: ', scipy.stats.fisher_exact(pd.crosstab(df.gender, df.late))[1])
display(pd.crosstab(df.gender, df.late))


# In[721]:


# Yellow: late; Blue: not late

colors_palette = {0: "blue", 1: "yellow", 2: "green"}
colors = [colors_palette[c] for c in df.late] 
_ = scatter_matrix(df.loc[:, ~df.columns.isin(['late'])], 
                   c = colors, figsize = [13,13], s = 150, grid = True, alpha = .7, diagonal='kde')

plt.show()


# In[750]:


# Dummy variable categorical variables in table

def rmcol(data, cols):
    return data.loc[:, ~data.columns.isin(cols)]

df_dum = pd.get_dummies(rmcol(df,['loan_status','days_overdue','late']))
df_dum.head()


# In[754]:


# Scale the data

df_scl = pd.DataFrame(preprocessing.scale(df_mod), columns = df_dum.columns.values)
df_scl.head()


# In[786]:


# Split data into train and test

X_train, X_test, y_train, y_test = train_test_split(df_scl, df.late, 
                                                    test_size=0.2, shuffle  = True, random_state = 123, stratify = df.late)
display(X_train.head(), y_train.head())


# In[787]:


display(X_train.shape, X_test.shape)


# In[858]:


# Explain late status using k Neighbors

myacc = np.empty(len(X_train))
mytr = np.empty(len(X_train))

for i in range(len(X_train)):
    knn = KNeighborsClassifier(n_neighbors = i + 1)
    knn.fit(X_train, y_train)
    myacc[i] = knn.score(X_test, y_test)
    mytr[i] = knn.score(X_train, y_train)
    

_ = plt.plot((np.arange(400) + 1), myacc)
_ = plt.plot((np.arange(400) + 1), mytr)
_ = plt.title('k-NN: Varying Number of Neighbors')
_ = plt.xlabel('Number of Neighbors')
_ = plt.ylabel('Accuracy')
_ = plt.legend(['Testing Accuracy','Training Accuracy'])
plt.show()

# Identify optimal k neighbors

loc_max = np.where(myacc == max(myacc))

print(
    "Accuracy: ", myacc[loc_max],
    "Number of Neighbors: ", (np.arange(400) + 1)[loc_max]
)


# In[859]:


# Check the independence between the independent variables, multicollinearity


f, ax = plt.subplots(figsize=(10, 8))
corr = pd.get_dummies(df.loc[:, ~df.columns.isin(['late', 'loan_status'])]).corr()
sns.heatmap(corr, vmin = -1, vmax = 1, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

display(corr)


# In[897]:


X_train, X_test, y_train, y_test = train_test_split(df_dum, df.late, 
                                                    test_size=0.2, shuffle  = True, random_state = 123, stratify = df.late)
display(X_train.head(), y_train.head())
X_train.info()


# In[943]:


classifier = LogisticRegression(random_state=123, solver = 'liblinear')
classifier.fit(X_train, y_train)

prob_score = classifier.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(np.array(y_test), prob_score, pos_label = 1)

y_pred = classifier.predict(X_test)

confusion_mx = metrics.confusion_matrix(np.array(y_test),np.array(y_pred))
print(confusion_mx)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

print(X_train.columns.values,classifier.coef_)


# In[944]:


roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

