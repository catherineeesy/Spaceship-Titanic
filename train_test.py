import pandas as pd 
import numpy as np 

#training dataset 
train = pd.read_csv("train.csv", encoding = "utf-8", index_col="PassengerId")
#replacing blanks with Nan
train.replace(r'^\s*$', np.nan, regex=True, inplace=True)

#separating cabin info to deck/number/port
splitting_cabin = train["Cabin"].str.split("/", expand = True) #expand=True tells pandas to create separate columns for each part.
train["Deck"] = splitting_cabin[0]
train["Port"] = splitting_cabin[2]

#making passengerId into a string 
making_passengerid_to_string =  train.index.astype(str)

#concatenating passenger Id by _
splitting_group_from_passengerId = making_passengerid_to_string.str.split("_", expand = True)
#extracting group and numer from passengerId
train["Group"] = splitting_group_from_passengerId.get_level_values(0)
train["PassengerNumber"] = splitting_group_from_passengerId.get_level_values(1)

#get group sizes
group_sizes = train.groupby("Group")["PassengerNumber"].count()
train["GroupSize"] = train["Group"].map(group_sizes)


#changing true and false to 1 and 0, respectively. 
train["CryoSleep"] = train["CryoSleep"].map({False: 0, True: 1}).astype("Int64")
train["Transported"] = train["Transported"].map({False: 0, True:1})
train["VIP"] = train["VIP"].map({False: 0, True:1})
train["HomePlanet"] = train["HomePlanet"].map({ "Europa": 1, "Earth": 2, "Mars": 3})
train["Destination"] = train["Destination"].map({ "TRAPPIST-1e": 1, "PSO J318.5-22": 2, "55 Cancri e": 3})
train["Deck"] = train["Deck"].map({"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "T":8}).astype("Int64")
train["Port"] = train["Port"].map({"P": 1, "S":2})

#combining columns to get total amount of spendings 
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

#for every column in cols 
for col in cols:
      #create another column in the dataset that has specific col_missing. if missing, row will have 1. 
      train[col + "_missing"] = train[col].isna().astype(int)
# Sum spending
train["Spending_in_Spaceship"] = train[cols].sum(axis=1, skipna=True)
# Drop original spending columns
train = train.drop(cols, axis=1)

#infers that if the spending is > 0, then peron is not in cryosleep
train.loc[(train["Spending_in_Spaceship"] > 0) & (train["CryoSleep"].isna()), "CryoSleep"] = 1

# Fill missing HomePlanet based on group
train['HomePlanet'] = train.groupby(train.index.str.split('_').str[0])['HomePlanet']\
                          .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

#dropping irrelevant columns 
train = train.drop(["Name", "Cabin", "PassengerNumber"], axis = 1)

#getting relationships 
cryo = train.groupby("CryoSleep")["Transported"].mean()
age = train.groupby("Age")["Transported"].mean()

#checking rows with nan
rows_with_na = train[train.isna().any(axis=1)]

#getting sum of nan values in each columns
checking_nan = train.isna().sum()


#cleaning test data
test = pd.read_csv("test.csv", encoding = "utf-8", index_col="PassengerId")
#replacing blanks with Nan
test.replace(r'^\s*$', np.nan, regex=True, inplace=True)

#separating cabin info to deck/number/port
splitting_cabin = test["Cabin"].str.split("/", expand = True) #expand=True tells pandas to create separate columns for each part.
test["Deck"] = splitting_cabin[0]
test["Port"] = splitting_cabin[2]

#making passengerId into a string 
making_passengerid_to_string =  test.index.astype(str)

#concatenating passenger Id by _
splitting_group_from_passengerId = making_passengerid_to_string.str.split("_", expand = True)
#extracting group and numer from passengerId
test["Group"] = splitting_group_from_passengerId.get_level_values(0)
test["PassengerNumber"] = splitting_group_from_passengerId.get_level_values(1)

#get group sizes
group_sizes = test.groupby("Group")["PassengerNumber"].count()
test["GroupSize"] = test["Group"].map(group_sizes)


#changing true and false to 1 and 0, respectively. 
test["CryoSleep"] = test["CryoSleep"].map({False: 0, True: 1}).astype("Int64")
test["VIP"] = test["VIP"].map({False: 0, True:1})
test["HomePlanet"] = test["HomePlanet"].map({ "Europa": 1, "Earth": 2, "Mars": 3})
test["Destination"] = test["Destination"].map({ "TRAPPIST-1e": 1, "PSO J318.5-22": 2, "55 Cancri e": 3})
test["Deck"] = test["Deck"].map({"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "T":8}).astype("Int64")
test["Port"] = test["Port"].map({"P": 1, "S":2})

#combining columns to get total amount of spendings 
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

#for every column in cols 
for col in cols:
      #create another column in the dataset that has specific col_missing. if missing, row will have 1. 
      test[col + "_missing"] = test[col].isna().astype(int)
# Sum spending
test["Spending_in_Spaceship"] = test[cols].sum(axis=1, skipna=True)
# Drop original spending columns
test = test.drop(cols, axis=1)

#infers that if the spending is > 0, then peron is not in cryosleep
test.loc[(train["Spending_in_Spaceship"] > 0) & (test["CryoSleep"].isna()), "CryoSleep"] = 1

# Fill missing HomePlanet based on group
test['HomePlanet'] = test.groupby(test.index.str.split('_').str[0])['HomePlanet']\
                          .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))


#dropping irrelevant columns 
test = test.drop(["Name", "Cabin", "PassengerNumber"], axis = 1)

checking_nan_test = train.isna().sum()
print(checking_nan_test)