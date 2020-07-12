# Linear_Regression_Project
Linear regression project in order to decide for a company whether to focus their efforts on a mobile app experience or a website.
## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Getting the Data

 I have worked with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
 - Avg. Session Length: Average session of in-store style advice sessions.
 - Time on App: Average time spent on App in minutes
 - Time on Website: Average time spent on Website in minutes
 - Length of Membership: How many years the customer has been a member.
 
 I read in the Ecommerce Customers csv file as a DataFrame called customers  and check the head of customers, and check out its info() and describe() methods for more information.
 
 ```python
 customers = pd.read_csv("Ecommerce Customers")
 customers.head()
 customers.describe()
 customers.info()
 ```

<img src= "https://user-images.githubusercontent.com/66487971/87251267-f2614c00-c472-11ea-93ff-6a2ddc6d7d50.png" width = 1000>
<img src= "https://user-images.githubusercontent.com/66487971/87251287-1fadfa00-c473-11ea-8ab3-2d0db298a2d7.png" width = 750>
<img src= "https://user-images.githubusercontent.com/66487971/87251319-774c6580-c473-11ea-9bfe-55ec30f8e847.png" width = 400>

## Exploratory Data Analysis
I use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.

```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
```

<img src= "https://user-images.githubusercontent.com/66487971/87251413-36a11c00-c474-11ea-8222-5f82d4810a1c.png" width = 500>

It's hard to see a correlation between time on website and yearly spent columns. I try the same for the time on app column.

```python
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
```

<img src= "https://user-images.githubusercontent.com/66487971/87251461-a1525780-c474-11ea-9eca-9e2dc2a21928.png" width = 500>

There seems to be a stronger correlation here.
Now I use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.

```python
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
```
<img src= "https://user-images.githubusercontent.com/66487971/87251542-263d7100-c475-11ea-9ff5-0e0650ab9757.png" width = 500>

I explore these types of relationships across the entire data set using pairplot.

```python
sns.pairplot(customers)
```
<img src= "https://user-images.githubusercontent.com/66487971/87251608-8207fa00-c475-11ea-9eb1-dc687bb975fe.png" width = 900>

Based off this plot Length of Membership  looks to be the most correlated feature with Yearly Amount Spent.
I create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.

```python
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
```

<img src= "https://user-images.githubusercontent.com/66487971/87251663-fe024200-c475-11ea-9d56-13a31425b55b.png" width = 500>

## Training and Testing Data

Now I split the data into training and testing sets. I set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.

```python
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
```

 Splitting the data into training and testing sets.
 
 ```python
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
 
 ```
 
 ## Training the Model
 
  Now I train my model on the training data.
  
  ```python
  
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  lm.fit(X_train,y_train)
  
  ```
  
  I print out the coefficients of the model.
  
  ```python
  print('Coefficients: \n', lm.coef_)
  ```
  
  <img src= "https://user-images.githubusercontent.com/66487971/87251820-2f2f4200-c477-11ea-9b8c-946ac78df08d.png" width = 200>
  
  Now that I have fit my model, now I evaluate its performance by predicting off the test values.
  
  ```python
  predictions = lm.predict( X_test)
  plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

```

<img src= "https://user-images.githubusercontent.com/66487971/87251865-7289b080-c477-11ea-84fe-df3f65b9fed4.png" width = 500>

My model did quite well!

## Evaluating the Model

Now I evaluate my model performance by calculating the residual sum of squares.

```python
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
<img src= "https://user-images.githubusercontent.com/66487971/87252012-8681e200-c478-11ea-8a92-7565c3bb91e1.png" width = 200>

## Residuals

I plot a histogram of the residuals and make sure it looks normally distributed.

```python
sns.distplot((y_test-predictions),bins=50);
```
<img src= "https://user-images.githubusercontent.com/66487971/87252059-eed0c380-c478-11ea-927a-aeb0d9baaa41.png" width = 350>

## Conclusion

I interpret the coefficients at all to get an idea to decide if the company should focus their efforts on mobile app or website development.

```python
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
```
<img src= "https://user-images.githubusercontent.com/66487971/87252168-a9f95c80-c479-11ea-8ab4-81cbec7cf8f5.png" width = 150>

Interpreting the coefficients:
- Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

This can be both interpered as : Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. 

## This concludes my project here. Thanks for reading all the way through.









  







 

