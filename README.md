
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
data = None
```

Now, display the dataset to ensure everything loaded correctly. 

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

This heatmap is useful for tracking unusual correlations. There is nothing really unexpected about this correlation matrix. The diagonal has correlation 1, which makes sense, and all the other correlations seem to be somewhere between 0 and 0.8. 

In order to perform a succesful PCA, you'd want to have some higher correlations among variables (which seems to be the case here, eg burger/bread and burger/tomatoes) so dimensionality reduction makes sense. If all variables would be uncorrelated, it would be hard to use PCA in order to reduce dimensionality. On the other hand, if variables are perfect correlates, you should just go ahead and remove columns instead of performing PCA.

## 3. Explore the data distributions

Let's use `.describe()` to get a sense of our data distributions.

Let's also plot some histograms of the distribution of our dataset.  

In the cell below, create a histogram of the `data`.  Pass in the following parameters:

* `bins=6`
* `xlabelsize=8`
* `ylabelsize=8`
* `figsize=(8,8)`


```python
ax = None
```

These distributions look approximately normal (note that there are only 23 observations, so it is to be expected that the curves are not perfectly bell-shaped!). 

Now, let's go ahead and standardize the data. We'll do this manually at first to understand what is going on in the process. 

### 3.1 standardize manually

In the cell below, compute the following values and store them in the appropriate variables.

1. The `mean` of the `data`.
1. The standard error of the `data`.
1. A standardized version of the dataset, where each value has had the `avg` subtracted, and then divided by the `std_err`.  


```python
avg = None
std_err = None
data_std= None
```

Now, display the head of the standardized dataset.

Finally, let's display histograms of this dataset, as we did with the original above.  Pass in the same parameters as you did above when creating these histograms.  


```python
ax = None
```

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

Note that you have to reattach the names of the columns.

Run the cell below to reattach the column names.


```python
data_std_2.columns = list(data)
```


```python
data_std_2.head()
```

Finally, create another set of histograms, this time on `data_std_2`.  Use the same parameters as we have above. 


```python
ax = None
```

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

Or, even easier, we can make use of the `.cov` function inside of numpy, and just pass in the transposed version of our data.  


```python
np.cov(data_std.T)
```

## 5.  Eigendecomposition in Python

In Python, numpy stores quite a few linear algebra functions which makes eigendecomposition easy, stored inside the `linalg` module.  

In the cell below, call `linalg.eig()` and pass in the covariance matrix we computed above, `cov_mat`.  

Note that this function returns 2 values:

1. An 1-d array of eigenvalues
1. A 2-d array of eigenvectors


```python
eig_values, eig_vectors = None
```

Now, let's inspect the eiginvalues and eigenvectors this function returned:


```python
eig_values
```


```python
eig_vectors
```

And finally, we'll use a list comprehension to compute the eigenpairs.  Run the cell below. 


```python
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]
```


```python
eig_pairs
```

### 5.1  Check if the squared norms are equal to 1


```python
for eigvec in eig_vectors:
    print(np.linalg.norm(eigvec))
```

or, alternatively


```python
sum(np.square(eig_vectors))
```

### 5.2 Let's check if our eigenvectors are uncorrelated. 

Run the cells below to create a correlation heatmap as we did at the top of the lab, but this time on a correlation matrix of the eigenvectors we've computed.  


```python
eig_vectors
```


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


```python
eig_vectors.corr()
```

Great, you got to the end of this lab! You know how to transform your data now. But what's the use and what does this all mean? You'll find out in the next lecture and lab!

## Sources

https://data.world/exercises/principal-components-exercise-1

https://data.world/craigkelly/usda-national-nutrient-db

https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
