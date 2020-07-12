# Linear_Regression_Project
Linear regression project in order to decide whether to focus their efforts on a mobile app experience or a website for a company.
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
<img src= "https://user-images.githubusercontent.com/66487971/87251608-8207fa00-c475-11ea-9eb1-dc687bb975fe.png" width = 800>






 

