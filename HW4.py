import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error as MSE
house_df=pd.read_csv('C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Week 4/housing2(2).csv')
print(house_df.head())
col_1=['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT','MEDV']
sns.pairplot(house_df[col_1],height=2.5)
plt.tight_layout()
plt.show()
corr=house_df[col_1].corr()
print(corr)
sns.set(font_scale=0.5)
sns.heatmap(corr,annot=True,)
plt.show()
print(("Number of Rows of Data = " + str(len(house_df)) + '\n'))
n_rows=len(house_df)
n_col=0
for column in house_df.values[0,:]:
    n_col=n_col+1
print("Number of columns of Data = " , n_col , '\n')
print("The summary for each column is \n",house_df.describe())
y=house_df['MEDV'].values
house_arr=house_df.values
X=house_arr[:,:26]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
print("the coefficients for the regression are ",lr.coef_)
#%%
y_labels=np.array(house_df.keys())
y_labels=y_labels[:26]
plt.scatter(y_labels,lr.coef_)
plt.show()
y_pred=lr.predict(X_test)
p_red_train=lr.predict(X_train)
print("The training set accuracy is ", lr.score(X_train,y_train))
print("The test set accuracy is ", lr.score(X_test,y_test))

#now standard scaling the test and training sets
ssc=StandardScaler()
X_ssc=ssc.fit_transform(X)
y_ssc=ssc.fit_transform(y[:,np.newaxis]).flatten()
X_train_ssc,X_test_ssc,y_train_ssc,y_test_ssc=train_test_split(X_ssc,y_ssc,test_size=0.3,random_state=21)
lr_ssc=LinearRegression()
lr_ssc.fit(X_train_ssc,y_train_ssc)
print("the coefficients for the regression are ",lr_ssc.coef_)

plt.scatter(y_labels,lr_ssc.coef_,s=2)
plt.xlabel('Attributes (Standardized')
plt.ylabel('Correlation coefficient (Standardized)')
plt.show()
y_pred_ssc=lr.predict(X_test_ssc).reshape(-1,1)
y_pred_train_ssc=lr.predict(X_train_ssc).reshape(-1,1)
print("The training set accuracy is (on standardization) ", lr_ssc.score(X_train_ssc,y_train_ssc))
print("The test set accuracy is (on standardization) ", lr_ssc.score(X_test_ssc,y_test_ssc))
print("The MSE is ", MSE(y_test,y_pred))

#%%
sns.set(font_scale=1)
plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)

for i in range(n_col-1):
    sns.regplot(X_train_ssc[:,i],y_train_ssc,ci=None,color='blue')
    plt.xlabel(y_labels[i])
    plt.ylabel('MEDV')
    plt.show()                


#%%
from sklearn.linear_model import Ridge
ridge = Ridge(alpha= 0.01, normalize= True)
ridge.fit(X_train_ssc,y_train_ssc)
y_pred_rid = ridge.predict(X_test_ssc)
y_pred_train_rid = ridge.predict(X_train_ssc)
print("The training set accuracy is (on standardization) for Ridge ", ridge.score(X_train_ssc,y_train_ssc))
print("The test set accuracy is (on standardization) for Ridge ", ridge.score(X_test_ssc,y_test_ssc))
print("The MSE is ", MSE(y_test_ssc,y_pred_rid))
testacc=[]
trainacc=[]
coef_r=[]
y_axis=[]
X_zero=[]
for i in range (1,20):
    X_zero.append(0)
for i in range (1,20):
    y_axis.append(i)
for i in range (1,20):
    ridge=Ridge(alpha = i, normalize=True)
    ridge.fit(X_train_ssc,y_train_ssc)
    coef_r.append(ridge.coef_)
    testacc.append(ridge.score(X_test_ssc,y_test_ssc))
    trainacc.append(ridge.score(X_train_ssc,y_train_ssc))
coef_r=np.array(coef_r)
print("the coefficients for the regression are ",coef_r)
plt.title(" Ridge Regression Model")
plt.plot(y_axis,coef_r)
plt.plot(y_axis,X_zero,color='black', lw=2)
plt.xlabel("Alpha Values")
plt.ylabel("Coefficients")
plt.legend(y_labels)
plt.show()
plt.plot()

plt.title("Accuracy vs alpha scores for Ridge Regression Model")
plt.plot(y_axis,testacc,label="Test Accuracy")
plt.plot(y_axis,trainacc,label="Train Accuracy")
plt.legend()
plt.xlabel('Alpha Values')
plt.ylabel('Acuracy')
plt.show()
#%%
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.0001,normalize=True)
lasso.fit(X_train_ssc,y_train_ssc)
y_pred_las = lasso.predict(X_test_ssc)
y_pred_train_las = lasso.predict(X_train_ssc)
print("The training set accuracy is (on standardization) for Lasso ", lasso.score(X_train_ssc,y_train_ssc))
print("The test set accuracy is (on standardization) for Lasso ", lasso.score(X_test_ssc,y_test_ssc))
print("The MSE is ", MSE(y_test_ssc,y_pred_las))
testacc_las=[]
trainacc_las=[]
coef_las=[]
y_axis=[]
X_zero=[]
for i in range (15,1000):
    X_zero.append(0)
for i in range (15,1000):
    y_axis.append(1/i)
for i in range(15,1000):
    lasso=Lasso(alpha = 1/i)
    lasso.fit(X_train_ssc,y_train_ssc)
    coef_las.append(lasso.coef_)
    testacc_las.append(lasso.score(X_test_ssc,y_test_ssc))
    trainacc_las.append(lasso.score(X_train_ssc,y_train_ssc))
coef_las=np.array(coef_las)
print("the coefficients for the regression are ",coef_las)
plt.title(" Lasso Regression Model")
plt.plot(y_axis,coef_las)
plt.plot(y_axis,X_zero,color='black', lw=2)
plt.xscale('log')
plt.xlabel("Alpha Values")
plt.ylabel("Coefficients")
plt.legend(y_labels)
plt.show()
plt.plot()

plt.title("Accuracy vs alpha scores for Lasso Regression Model")
plt.plot(y_axis,testacc_las,label="Test Accuracy")
plt.plot(y_axis,trainacc_las,label="Train Accuracy")
plt.legend()
plt.xlabel('Alpha Values')
plt.ylabel('Acuracy')
plt.show()


#%%

def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=4)
X_dtr=X[:,18]
dtr.fit(X_dtr.reshape(-1,1),y)
sort_idx=X_dtr.flatten().argsort()
lin_regplot(X_dtr[sort_idx].reshape(-1,1), y[sort_idx], dtr)
plt.title("Decision Tree regression for Rooms vs MEDV")
plt.ylabel("MEDV")
plt.xlabel("Average Number of rooms per dwelling")
plt.show()

print("My name is Vedant Mundada")
print("My NetID is: vkm3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






















