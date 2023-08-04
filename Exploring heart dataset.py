import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Kaggle dataset 
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

(df.columns)   # printing features 

# Exploring data
import seaborn as sns
sns.pairplot(df)   


df["DEATH_EVENT"].value_counts()          # counting total deaths and alive from deathcount column

# Bar graph
df["DEATH_EVENT"].value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
plt.xlabel("DEATH COUNT")
plt.ylabel("COUNT")

df.isna().sum()  # checking for null values


# HEART FAILURE FREQUENCY ON THE BASICS OF GENDER
df.rename(columns = {"sex": "gender"} , inplace= True)
df["gender"].value_counts()
#crosstab values 
pd.crosstab(df.DEATH_EVENT , df.gender)


# HEART FAILURE FREQUENCY ON THE BASICS OF AGE
df["age"].value_counts().head()

#crosstab values 
pd.crosstab(df.DEATH_EVENT , df.age)

#scatterplot to determine deaths on basics of Age

plt.scatter(df.gender[df.DEATH_EVENT==1] , df.age[df.DEATH_EVENT==1] ,c="salmon" )
plt.title("Increasing Deaths due to Heart failure")
plt.xlabel("0=MALE      |     1=FEMALE")
plt.ylabel(" AGE ")

