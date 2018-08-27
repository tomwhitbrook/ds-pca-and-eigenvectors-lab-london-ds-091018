
# Principal Components Analysis lab

## Objective

In this lab, we'll use PCA to transform a dataset by computing a covariance matrix and performing eigendecomposition.

## 1. Introduction and data

Let's ease into the concept of Principal Component Analysis by looking at a first data set. This first data set contains the Estimated Retail Prices by Cities in March 1973. The numbers listed are the average price by cents in pounds. 


```python
import pandas as pd
import numpy as np
#from sklearn.decomposition import PCA
#import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
% matplotlib inline
```

In the cell below, read in the data from `foodusa.csv` and store in in a pandas DataFrame.  Be sure to set `index_col` to `0`.


```python
data = pd.read_csv('foodusa.csv', index_col=0)
```

Now, display the dataset to ensure everything loaded correctly. 


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>24.5</td>
      <td>94.5</td>
      <td>73.9</td>
      <td>80.1</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>26.5</td>
      <td>91.0</td>
      <td>67.5</td>
      <td>74.6</td>
      <td>53.3</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>29.7</td>
      <td>100.8</td>
      <td>61.4</td>
      <td>104.0</td>
      <td>59.6</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>22.8</td>
      <td>86.6</td>
      <td>65.3</td>
      <td>118.4</td>
      <td>51.2</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>26.7</td>
      <td>86.7</td>
      <td>62.7</td>
      <td>105.9</td>
      <td>51.2</td>
    </tr>
    <tr>
      <th>CINCINNATI</th>
      <td>25.3</td>
      <td>102.5</td>
      <td>63.3</td>
      <td>99.3</td>
      <td>45.6</td>
    </tr>
    <tr>
      <th>CLEVELAND</th>
      <td>22.8</td>
      <td>88.8</td>
      <td>52.4</td>
      <td>110.9</td>
      <td>46.8</td>
    </tr>
    <tr>
      <th>DALLAS</th>
      <td>23.3</td>
      <td>85.5</td>
      <td>62.5</td>
      <td>117.9</td>
      <td>41.8</td>
    </tr>
    <tr>
      <th>DETROIT</th>
      <td>24.1</td>
      <td>93.7</td>
      <td>51.5</td>
      <td>109.7</td>
      <td>52.4</td>
    </tr>
    <tr>
      <th>HONALULU</th>
      <td>29.3</td>
      <td>105.9</td>
      <td>80.2</td>
      <td>133.2</td>
      <td>61.7</td>
    </tr>
    <tr>
      <th>HOUSTON</th>
      <td>22.3</td>
      <td>83.6</td>
      <td>67.8</td>
      <td>108.6</td>
      <td>42.4</td>
    </tr>
    <tr>
      <th>KANSAS CITY</th>
      <td>26.1</td>
      <td>88.9</td>
      <td>65.4</td>
      <td>100.9</td>
      <td>43.2</td>
    </tr>
    <tr>
      <th>LOS ANGELES</th>
      <td>26.9</td>
      <td>89.3</td>
      <td>56.2</td>
      <td>82.7</td>
      <td>38.4</td>
    </tr>
    <tr>
      <th>MILWAUKEE</th>
      <td>20.3</td>
      <td>89.6</td>
      <td>53.8</td>
      <td>111.8</td>
      <td>53.9</td>
    </tr>
    <tr>
      <th>MINNEAPOLIS</th>
      <td>24.6</td>
      <td>92.2</td>
      <td>51.9</td>
      <td>106.0</td>
      <td>50.7</td>
    </tr>
    <tr>
      <th>NEW YORK</th>
      <td>30.8</td>
      <td>110.7</td>
      <td>66.0</td>
      <td>107.3</td>
      <td>62.6</td>
    </tr>
    <tr>
      <th>PHILADELPHIA</th>
      <td>24.5</td>
      <td>92.3</td>
      <td>66.7</td>
      <td>98.0</td>
      <td>61.7</td>
    </tr>
    <tr>
      <th>PITTSBURGH</th>
      <td>26.2</td>
      <td>95.4</td>
      <td>60.2</td>
      <td>117.1</td>
      <td>49.3</td>
    </tr>
    <tr>
      <th>ST LOUIS</th>
      <td>26.5</td>
      <td>92.4</td>
      <td>60.8</td>
      <td>115.1</td>
      <td>46.2</td>
    </tr>
    <tr>
      <th>SAN DIEGO</th>
      <td>25.5</td>
      <td>83.7</td>
      <td>57.0</td>
      <td>92.8</td>
      <td>35.4</td>
    </tr>
    <tr>
      <th>SAN FRANCISCO</th>
      <td>26.3</td>
      <td>87.1</td>
      <td>58.3</td>
      <td>101.8</td>
      <td>41.5</td>
    </tr>
    <tr>
      <th>SEATTLE</th>
      <td>22.5</td>
      <td>77.7</td>
      <td>62.0</td>
      <td>91.1</td>
      <td>44.9</td>
    </tr>
    <tr>
      <th>WASHINGTON DC</th>
      <td>24.2</td>
      <td>93.8</td>
      <td>66.0</td>
      <td>81.6</td>
      <td>46.2</td>
    </tr>
  </tbody>
</table>
</div>



As you know from the lecture, we'll perform an eigendecomposition to transform the data. 
Before doing so, two things are of main importance when 
We'll first perform 2 steps:

1. Let's look at the current correlation structure of the data and see if anything stands out.
2. Let's also look at the data distributions and make sure we standardize our data first. As mentioned, we would like to get our data on a unit scale, if we want to get optimal PCA performance in the long run. 

## 2. The correlation structure

Before we start, let's have a quick exploratory look at the data. Let's generate the correlation heatmap to see if we can detect anything extraordinary.

Run the cells below to create a Correlation Heatmap of our dataset. 


```python
names = list(data)
```


```python
fig, ax = plt.subplots(figsize=(5,5))
mat = ax.matshow(data.corr())
ax.set_xticks(np.arange(0,5,1))
ax.set_yticks(np.arange(0,5,1))
ax.set_xticklabels(names, rotation = 45)
ax.set_yticklabels(names)
fig.colorbar(mat)
plt.show();
```


![png](output_12_0.png)


This heatmap is useful for tracking unusual correlations. There is nothing really unexpected about this correlation matrix. The diagonal has correlation 1, which makes sense, and all the other correlations seem to be somewhere between 0 and 0.8. 

In order to perform a succesful PCA, you'd want to have some higher correlations among variables (which seems to be the case here, eg burger/bread and burger/tomatoes) so dimensionality reduction makes sense. If all variables would be uncorrelated, it would be hard to use PCA in order to reduce dimensionality. On the other hand, if variables are perfect correlates, you should just go ahead and remove columns instead of performing PCA.

## 3. Explore the data distributions

Let's use `.describe()` to get a sense of our data distributions.


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.291304</td>
      <td>91.856522</td>
      <td>62.295652</td>
      <td>102.991304</td>
      <td>48.765217</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.506884</td>
      <td>7.554940</td>
      <td>6.950244</td>
      <td>14.239252</td>
      <td>7.602668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.300000</td>
      <td>77.700000</td>
      <td>51.500000</td>
      <td>74.600000</td>
      <td>35.400000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.700000</td>
      <td>86.900000</td>
      <td>57.650000</td>
      <td>95.400000</td>
      <td>42.800000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25.300000</td>
      <td>91.000000</td>
      <td>62.500000</td>
      <td>105.900000</td>
      <td>46.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.500000</td>
      <td>94.150000</td>
      <td>66.000000</td>
      <td>111.350000</td>
      <td>52.850000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.800000</td>
      <td>110.700000</td>
      <td>80.200000</td>
      <td>133.200000</td>
      <td>62.600000</td>
    </tr>
  </tbody>
</table>
</div>



Let's also plot some histograms of the distribution of our dataset.  

In the cell below, create a histogram of the `data`.  Pass in the following parameters:

* `bins=6`
* `xlabelsize=8`
* `ylabelsize=8`
* `figsize=(8,8)`


```python
ax = data.hist(bins=6, xlabelsize=8, ylabelsize= 8, figsize=(8,8))
```


![png](output_18_0.png)


These distributions look approximately normal (note that there are only 23 observations, so it is to be expected that the curves are not perfectly bell-shaped!). 

Now, let's go ahead and standardize the data. We'll do this manually at first to understand what is going on in the process. 

### 3.1 standardize manually

In the cell below, compute the following values and store them in the appropriate variables.

1. The `mean` of the `data`.
1. The standard error of the `data`.
1. A standardized version of the dataset, where each value has had the `avg` subtracted, and then divided by the `std_err`.  


```python
avg = data.mean()
std_err = data.std()
data_std= (data - avg)/ std_err
```

Now, display the head of the standardized dataset.


```python
data_std.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>-0.315653</td>
      <td>0.349901</td>
      <td>1.669632</td>
      <td>-1.607620</td>
      <td>-0.942461</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>0.482151</td>
      <td>-0.113372</td>
      <td>0.748801</td>
      <td>-1.993876</td>
      <td>0.596473</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>1.758636</td>
      <td>1.183792</td>
      <td>-0.128866</td>
      <td>0.070839</td>
      <td>1.425129</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>-0.993785</td>
      <td>-0.695773</td>
      <td>0.432265</td>
      <td>1.082128</td>
      <td>0.320254</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>0.561931</td>
      <td>-0.682536</td>
      <td>0.058178</td>
      <td>0.204273</td>
      <td>0.320254</td>
    </tr>
  </tbody>
</table>
</div>



Finally, let's display histograms of this dataset, as we did with the original above.  Pass in the same parameters as you did above when creating these histograms.  


```python
ax = data_std.hist(bins=6, xlabelsize=8, ylabelsize= 8, figsize=(8,8))
```


![png](output_25_0.png)


It seems like nothing really changed here, but be aware of what happened to the x-axis!

### 3.2 Another way to standardize the data

Since this is a common operation, sklearn provides an easy way to scale and transform our dataset by using a `StandardScaler` object's `fit_transform()` method.

Run the cell below to use sklearn to create a scaled version of the dataset, and then inspect the head of this new DataFrame to see how it compares to the dataset we scaled manually.


```python
from sklearn.preprocessing import StandardScaler
data_std_2 = StandardScaler().fit_transform(data)
data_std_2 = pd.DataFrame(data_std_2)
```


```python
data_std_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.322747</td>
      <td>0.357765</td>
      <td>1.707156</td>
      <td>-1.643751</td>
      <td>-0.963643</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.492987</td>
      <td>-0.115920</td>
      <td>0.765630</td>
      <td>-2.038688</td>
      <td>0.609878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.798161</td>
      <td>1.210398</td>
      <td>-0.131763</td>
      <td>0.072431</td>
      <td>1.457158</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.016120</td>
      <td>-0.711410</td>
      <td>0.441980</td>
      <td>1.106449</td>
      <td>0.327451</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.574560</td>
      <td>-0.697876</td>
      <td>0.059485</td>
      <td>0.208864</td>
      <td>0.327451</td>
    </tr>
  </tbody>
</table>
</div>



Note that you have to reattach the names of the columns.

Run the cell below to reattach the column names.


```python
data_std_2.columns = list(data)
```


```python
data_std_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.322747</td>
      <td>0.357765</td>
      <td>1.707156</td>
      <td>-1.643751</td>
      <td>-0.963643</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.492987</td>
      <td>-0.115920</td>
      <td>0.765630</td>
      <td>-2.038688</td>
      <td>0.609878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.798161</td>
      <td>1.210398</td>
      <td>-0.131763</td>
      <td>0.072431</td>
      <td>1.457158</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.016120</td>
      <td>-0.711410</td>
      <td>0.441980</td>
      <td>1.106449</td>
      <td>0.327451</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.574560</td>
      <td>-0.697876</td>
      <td>0.059485</td>
      <td>0.208864</td>
      <td>0.327451</td>
    </tr>
  </tbody>
</table>
</div>



Finally, create another set of histograms, this time on `data_std_2`.  Use the same parameters as we have above. 


```python
ax = data_std_2.hist(bins=6, xlabelsize= 8, ylabelsize= 8 , figsize=(8,8))
```


![png](output_34_0.png)


Note that the results differ slightly. When using StandardScaler, centering and scaling happen independently on each feature by computing the relevant statistics. Mean and standard deviation are then stored to be used on later data using the transform method. You can look at the histograms again, but it is expected to look exactly the same again.

## 4. The correlation matrix

**_NOTE:_** This section contains **_a lot_** of math.  Understanding how the math behind PCA and eigendecomposition works is important, because it provides great insight  into how the algorithm actually works under the hood, and can inform our use to make sure we're using PCA correctly. 

With that being said, don't feel overwhelmed--on the job, you'll almost never compute this manually. Instead, you'll rely on heavily optimized, industry-standard tools such as sklearn to transform data with PCA.  

We've actually looked at the heatmap already, but now let's formally compute the correlation matrix. As you saw in the lecture, the sample covariance matrix is given by:

\begin{equation}
\mathbf{S} = \begin{bmatrix}
    s^2_{1} & s_{12}  & \dots  & s_{1p} \\
    s_{21} & s^2_{2}  & \dots  & s_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    s_{p1} & s_{p2} & \dots  & s^2_{p}
\end{bmatrix}
\end{equation}

with

$$s_{jk} = \dfrac{\sum_{i=1}^n (X_{ij}-\bar X_{.j})(X_{ij}-\bar X_{.k})}{n-1}= \dfrac{\sum_{i=1}^n x_{ij}x_{ik}}{n-1}$$

Everything became actually easier now that we're working with standardized variables here, so we can get to the correlations as follows:
$r_{jk} = \dfrac{\sum_{i=1}^n z_{ij}z_{ik}}{n-1}$.

We know that we can use the .corr-function, but it's a good exercise to do this manually in Python. You can use the `.dot`-function:


```python
cov_mat = (data_std.T.dot(data_std)) / (data_std.shape[0]-1)
```


```python
cov_mat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bread</th>
      <td>1.000000</td>
      <td>0.681700</td>
      <td>0.328239</td>
      <td>0.036709</td>
      <td>0.382241</td>
    </tr>
    <tr>
      <th>Burger</th>
      <td>0.681700</td>
      <td>1.000000</td>
      <td>0.333422</td>
      <td>0.210937</td>
      <td>0.631898</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>0.328239</td>
      <td>0.333422</td>
      <td>1.000000</td>
      <td>-0.002779</td>
      <td>0.254417</td>
    </tr>
    <tr>
      <th>Oranges</th>
      <td>0.036709</td>
      <td>0.210937</td>
      <td>-0.002779</td>
      <td>1.000000</td>
      <td>0.358061</td>
    </tr>
    <tr>
      <th>Tomatoes</th>
      <td>0.382241</td>
      <td>0.631898</td>
      <td>0.254417</td>
      <td>0.358061</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Or, even easier, we can make use of the `.cov` function inside of numpy, and just pass in the transposed version of our data.  


```python
np.cov(data_std.T)
```




    array([[ 1.        ,  0.68170049,  0.32823868,  0.03670916,  0.38224129],
           [ 0.68170049,  1.        ,  0.33342167,  0.21093676,  0.63189831],
           [ 0.32823868,  0.33342167,  1.        , -0.00277912,  0.2544167 ],
           [ 0.03670916,  0.21093676, -0.00277912,  1.        ,  0.3580615 ],
           [ 0.38224129,  0.63189831,  0.2544167 ,  0.3580615 ,  1.        ]])



## 5.  Eigendecomposition in Python

In Python, numpy stores quite a few linear algebra functions which makes eigendecomposition easy, stored inside the `linalg` module.  

In the cell below, call `linalg.eig()` and pass in the covariance matrix we computed above, `cov_mat`.  

Note that this function returns 2 values:

1. An 1-d array of eigenvalues
1. A 2-d array of eigenvectors


```python
eig_values, eig_vectors = np.linalg.eig(cov_mat)
```

Now, let's inspect the eiginvalues and eigenvectors this function returned:


```python
eig_values
```




    array([2.42246795, 1.10467489, 0.2407653 , 0.73848053, 0.49361132])




```python
eig_vectors
```




    array([[ 0.49614868,  0.30861972,  0.49989887,  0.38639398, -0.50930459],
           [ 0.57570231,  0.04380176, -0.77263501,  0.26247227,  0.02813712],
           [ 0.33956956,  0.43080905, -0.00788224, -0.83463952, -0.0491    ],
           [ 0.22498981, -0.79677694,  0.0059668 , -0.29160659, -0.47901574],
           [ 0.50643404, -0.28702846,  0.39120139,  0.01226602,  0.71270629]])



And finally, we'll use a list comprehension to compute the eigenpairs.  Run the cell below. 


```python
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]
```


```python
eig_pairs
```




    [(2.4224679507118134,
      array([0.49614868, 0.57570231, 0.33956956, 0.22498981, 0.50643404])),
     (1.1046748929130334,
      array([ 0.30861972,  0.04380176,  0.43080905, -0.79677694, -0.28702846])),
     (0.24076530483587502,
      array([ 0.49989887, -0.77263501, -0.00788224,  0.0059668 ,  0.39120139])),
     (0.738480531992392,
      array([ 0.38639398,  0.26247227, -0.83463952, -0.29160659,  0.01226602])),
     (0.49361131954688525,
      array([-0.50930459,  0.02813712, -0.0491    , -0.47901574,  0.71270629]))]



### 5.1  Check if the squared norms are equal to 1


```python
for eigvec in eig_vectors:
    print(np.linalg.norm(eigvec))
```

    0.9999999999999998
    1.0
    1.0
    0.9999999999999999
    1.0000000000000002
    

or, alternatively


```python
sum(np.square(eig_vectors))
```




    array([1., 1., 1., 1., 1.])



### 5.2 Let's check if our eigenvectors are uncorrelated. 

Run the cells below to create a correlation heatmap as we did at the top of the lab, but this time on a correlation matrix of the eigenvectors we've computed.  


```python
eig_vectors
```




    array([[ 0.49614868,  0.30861972,  0.49989887,  0.38639398, -0.50930459],
           [ 0.57570231,  0.04380176, -0.77263501,  0.26247227,  0.02813712],
           [ 0.33956956,  0.43080905, -0.00788224, -0.83463952, -0.0491    ],
           [ 0.22498981, -0.79677694,  0.0059668 , -0.29160659, -0.47901574],
           [ 0.50643404, -0.28702846,  0.39120139,  0.01226602,  0.71270629]])




```python
eig_vectors = pd.DataFrame(eig_vectors)
```


```python
fig, ax = plt.subplots(figsize=(5,5))
mat = ax.matshow(eig_vectors.corr())
ax.set_xticks(np.arange(0,5,1))
ax.set_yticks(np.arange(0,5,1))
ax.set_xticklabels(names, rotation = 45)
ax.set_yticklabels(names)
fig.colorbar(mat)
plt.show();
```


![png](output_59_0.png)



```python
eig_vectors.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.454959</td>
      <td>-0.175050</td>
      <td>0.713220</td>
      <td>0.448798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.454959</td>
      <td>1.000000</td>
      <td>0.007080</td>
      <td>-0.028847</td>
      <td>-0.018152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.175050</td>
      <td>0.007080</td>
      <td>1.000000</td>
      <td>0.011099</td>
      <td>0.006984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.713220</td>
      <td>-0.028847</td>
      <td>0.011099</td>
      <td>1.000000</td>
      <td>-0.028457</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.448798</td>
      <td>-0.018152</td>
      <td>0.006984</td>
      <td>-0.028457</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Great, you got to the end of this lab! You know how to transform your data now. But what's the use and what does this all mean? You'll find out in the next lecture and lab!

## Sources

https://data.world/exercises/principal-components-exercise-1

https://data.world/craigkelly/usda-national-nutrient-db

https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
