import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\mishr\\Downloads\\Housing.csv")
print(data.head(5))

#The Boston Housing dataset includes the following features:

#CRIM: Per capita crime rate by town.
#ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
#INDUS: Proportion of non-retail business acres per town.
#CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
#NOX: Nitric oxides concentration (parts per 10 million).
#RM: Average number of rooms per dwelling.
#AGE: Proportion of owner-occupied units built before 1940.
#DIS: Weighted distances to five Boston employment centers.
#RAD: Index of accessibility to radial highways.
#TAX: Full-value property tax rate per $10,000.
#PTRATIO: Pupil-teacher ratio by town.
#B: 1000(Bk - 0.63)^2, where Bk is the proportion of Black people by town.
#LSTAT: Percentage of lower status of the population.
#MEDV: Median value of owner-occupied homes in $1000s (target variable).

print(data.info())
print(data['CHAS'].value_counts())
print(data.describe()) 
# train-test split
x=data.iloc[:,:-1]
y=data['MEDV']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_train),len(x_test))
#To correct the value of #CHAS so that only one value is not present in the T-test.
from sklearn.model_selection import StratifiedShuffleSplit
ss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in ss.split(data,data['CHAS']):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]

print(strat_test_set['CHAS'].value_counts())   
print(strat_train_set['CHAS'].value_counts())
sns.heatmap(data=data.corr(),annot=True)
plt.show()
corr_matrix=data.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))  
from pandas.plotting import scatter_matrix
attributes=['MEDV','RM','ZN']
scatter_matrix(data[attributes],figsize=(12,8))
plt.show()
data.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8) 
plt.show()
# creating pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # Step 1: Fill missing values
    ('scaler', StandardScaler()),                  # Step 2: Scale features   # Step 3: Train model                                                 
])
data_num=my_pipeline.fit_transform(x_train) #the pipeline is designed to preprocess only the features (x_train).
print(data_num) #it is a numpy_array       #The labels (y_train) are passed directly to the model during training.
print(data_num.shape)
#selecting desired model
'''from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(data_num,y_train)
some_data=x_train.iloc[:5]
some_labels=y_train.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_labels))
#evaluating model
from sklearn.metrics import mean_squared_error
data_predictions=model.predict(data_num)
mse=mean_squared_error(y_train,data_predictions)
rmse=np.sqrt(mse)
print(rmse)
from sklearn.model_selection import cross_val_score
p=cross_val_score(LinearRegression(),data_num,y_train,scoring='neg_mean_squared_error',cv=5)  
rmse_scores=np.sqrt(-p)
print(rmse-rmse_scores)
def print_scores(scores):
    print('scores:',scores)
    print('mean:',scores.mean())
    print('standard deviation:',scores.std())
print(print_scores(rmse_scores)) '''
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(data_num,y_train)
some_data=x_train.iloc[:5]
some_labels=y_train.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_labels))
#evaluating model
from sklearn.metrics import mean_squared_error
data_predictions=model.predict(data_num)
mse=mean_squared_error(y_train,data_predictions)
rmse=np.sqrt(mse)
print(rmse)
from sklearn.model_selection import cross_val_score
p=cross_val_score(RandomForestRegressor(),data_num,y_train,scoring='neg_mean_squared_error',cv=5)   
rmse_scores=np.sqrt(-p)
print(rmse-rmse_scores)
def print_scores(scores):
    print('scores:',scores)
    print('mean:',scores.mean())
    print('standard deviation:',scores.std())
print(print_scores(rmse_scores))    # least mean error and standard deviation in all the 3 models'''
'''from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(data_num,y_train)
some_data=x_train.iloc[:5]
some_labels=y_train.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_labels))
#evaluating model
from sklearn.metrics import mean_squared_error
data_predictions=model.predict(data_num)
mse=mean_squared_error(y_train,data_predictions)
rmse=np.sqrt(mse)
print(rmse)
from sklearn.model_selection import cross_val_score 
p=cross_val_score(DecisionTreeRegressor(),data_num,y_train,scoring='neg_mean_squared_error',cv=5)  
rmse_scores=np.sqrt(-p)
print(rmse-rmse_scores)
def print_scores(scores):
    print('scores:',scores)
    print('mean:',scores.mean())
    print('standard deviation:',scores.std())
print(print_scores(rmse_scores)) '''
# SO BEST MODEL IS RANDOM FOREST REGRESSOR
#Testing the model on test data
x_test=strat_test_set.drop('MEDV',axis=1)
y_test=strat_test_set['MEDV'].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_prediction=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_prediction)
final_rmse=np.sqrt(final_mse)
print(final_rmse) 
# Using the model
print(prepared_data[0])
feature=np.array([[ 1.28770177, -0.50032012 , 1.03323679, -0.27808871 , 0.48925206 ,-1.42806858,
  1.02801516, -0.80217296 , 1.70689143 , 1.57843444 , 0.84534281 ,-0.07433689,
  1.75350503]])
print(model.predict(feature))
# Saving prediction and  Actual Values to a Dataframe
output_data = pd.DataFrame({
    'Feature1': x_test.iloc[:, 0],  # Replace with actual column names if required
    'Feature2': x_test.iloc[:, 1],
    'Feature3':x_test.iloc[:,2],
    'Feature4':x_test.iloc[:,3],
    'Feature5':x_test.iloc[:,4],
    'Feature6':x_test.iloc[:,5],
    'Feature7':x_test.iloc[:,6],
    'Feature8':x_test.iloc[:,7],
    'Feature9':x_test.iloc[:,8],
    'Feature10':x_test.iloc[:,9],
    'Feature11':x_test.iloc[:,10],
    'Feature12':x_test.iloc[:,11],
    'Feature13':x_test.iloc[:,12],
    'Actual MEDV': y_test,
    'Predicted MEDV': final_prediction
})

# Save DataFrame to a CSV file
#output_data.to_csv('C:\\Users\\mishr\\Downloads\\predictions.csv', index=False)

# Confirmation message
print("Predictions saved to 'predictions.csv'")
