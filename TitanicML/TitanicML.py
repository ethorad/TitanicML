import numpy as np
import pandas as pd

# define functions
def print_blank_count(df, col):
    total_rows = len(df)
    blanks = all_df[col].isna().sum()
    print("Blanks: " + str(blanks) + " (" + str(round(blanks/total_rows*100,2))+"%)")
    

print("*** Reading in data")
train_df = pd.read_csv("Data/TrainData.txt")
test_df = pd.read_csv("Data/TestData.txt")
all_df = pd.concat([train_df.drop(columns=["Survived"]),test_df])
print()

print("*** Summary of aggregate data")
all_df.info()
print()

print("*** Cleaning data")    

#PassengerId
column = "PassengerId"
print("Col: "+column)
print_blank_count(all_df, column)
print("-> OK, using unchanged")
print()

#Pclass
column = "Pclass"
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> OK, using unchanged")
print()

#Name
column = "Name"
print("Col: "+column)
print_blank_count(all_df, column)
print("-> OK, using unchanged")
print()

#Sex
column = "Sex"
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> Changing to an IsMale field")
column="IsMale"
train_df.rename(columns={"Sex":"IsMale"},inplace=True)
test_df.rename(columns={"Sex":"IsMale"},inplace=True)
all_df.rename(columns={"Sex":"IsMale"},inplace=True)
train_df[column].replace({"female":0,"male":1}, inplace=True)
test_df[column].replace({"female":0,"male":1}, inplace=True)
all_df[column].replace({"female":0,"male":1}, inplace=True)
print(all_df[column].value_counts())
print()

#Age
column="Age"
print("Col: "+column)
print_blank_count(all_df, column)
print("Median: " + str(round(all_df[column].median(),1)))
print("Mean: " + str(round(all_df[column].mean(),1)))
print("Std Dev: "+str(round(all_df[column].std(),1)))
print("-> Adding column for MissingAge flag")
print("-> And adding a column for IsAdult for age > 16")
print("-> And setting missing ages to the median")
print("-> Then normalise using mean / std dev")
# missing age flag
train_df["MissingAge"] = np.where(train_df["Age"].isna(), 1, 0)
test_df["MissingAge"] = np.where(test_df["Age"].isna(), 1, 0)
all_df["MissingAge"] = np.where(all_df["Age"].isna(), 1, 0)
# is adult
train_df["IsAdult"] = np.where(train_df["Age"]>16, 1, 0)
test_df["IsAdult"] = np.where(test_df["Age"]>16, 1, 0)
all_df["IsAdult"] = np.where(all_df["Age"]>16, 1, 0)
# replace blanks with median on all data
med = all_df[column].median()
train_df[column].fillna(med, inplace=True)
test_df[column].fillna(med, inplace=True)
all_df[column].fillna(med, inplace=True)
# normalise
mean = all_df[column].mean()
stddev = all_df[column].std()
train_df[column] = (train_df[column]-mean)/stddev
test_df[column] = (test_df[column]-mean)/stddev
all_df[column] = (all_df[column]-mean)/stddev
print("Median: " + str(round(all_df[column].median(),1)))
print("Mean: " + str(round(all_df[column].mean(),1)))
print("Std Dev: "+str(round(all_df[column].std(),1)))
print()

# SibSp
column="SibSp"
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> OK, using unchanged")
print()

# Parch
column="Parch"
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> OK, using unchanged")
print()

# Family group
print("-> Add new column Family combining SibSp and Parch")
column="Family"
train_df["Family"] = train_df["SibSp"] + train_df["Parch"]
test_df["Family"] = test_df["SibSp"] + test_df["Parch"]
all_df["Family"] = all_df["SibSp"] + all_df["Parch"]
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> OK, using unchanged")
print()

# Ticket
column="Ticket"
print("Col: "+column)
print_blank_count(all_df, column)
print("Unique count: "+str(all_df[column].nunique()) + " (" + str(round(all_df[column].nunique() / len(all_df)* 100,1)) + "%)")
print("-> Too many unique values, drop this column")
train_df.drop(columns=[column], inplace=True)
test_df.drop(columns=[column], inplace=True)
all_df.drop(columns=[column], inplace=True)
print()

#Fare
column ="Fare"
print("Col: "+column)
print_blank_count(all_df, column)
print("Median: " + str(round(all_df[column].median(),1)))
print("Mean: " + str(round(all_df[column].mean(),1)))
print("Std Dev: "+str(round(all_df[column].std(),1)))
print("-> OK, set blanks to median then normalise")
# replace blanks with median on all data
med = all_df[column].median()
train_df[column].fillna(med, inplace=True)
test_df[column].fillna(med, inplace=True)
all_df[column].fillna(med, inplace=True)
# normalise
mean = all_df[column].mean()
stddev = all_df[column].std()
train_df[column] = (train_df[column]-mean)/stddev
test_df[column] = (test_df[column]-mean)/stddev
all_df[column] = (all_df[column]-mean)/stddev
print("Median: " + str(round(all_df[column].median(),1)))
print("Mean: " + str(round(all_df[column].mean(),1)))
print("Std Dev: "+str(round(all_df[column].std(),1)))
print()

#Cabin
column ="Cabin"
print("Col: "+column)
print_blank_count(all_df, column)
print("Unique count: "+str(all_df[column].nunique()) + " (" + str(round(all_df[column].nunique() / len(all_df)* 100,1)) + "%)")
print("-> Too many blanks an unique values, drop column")
train_df.drop(columns=[column],  inplace=True)
test_df.drop(columns=[column],  inplace=True)
all_df.drop(columns=[column],  inplace=True)

#Embarked
column="Embarked"
print("Col: "+column)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print("-> Only two blanks, so replace with mode value")
mode = all_df[column].mode().iloc[0] # need iloc[0] since mode() returns a data frame with a single value
print("Mode: " + mode)
train_df[column].fillna(mode, inplace=True)
test_df[column].fillna(mode, inplace=True)
all_df[column].fillna(mode, inplace=True)
print_blank_count(all_df, column)
print(all_df[column].value_counts())
print()

print("Data cleaning complete")
all_df.info()
print()

print("Start machine learning")

print()
print("DONE")

