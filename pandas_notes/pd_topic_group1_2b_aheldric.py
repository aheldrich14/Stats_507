#!/usr/bin/env python3

# importing packages
from IPython.display import HTML
import pandas as pd 
import numpy as np
import os 
from scipy import stats
from scipy.stats import chi2_contingency
from collections import defaultdict
from scipy.stats import norm, inom, beta
import re 

# Andrew Heldrich
# aheldric@umich.edu

# ## `rank()` Method
# - A common need is to rank rows of data by position in a group
# - SQL has nice partitioning functions to accomplish this, e.g. `ROW_NUMBER()`
# - Pandas `rank()` can be used to achieve similar results

# ## Example
# - If we have sales data for individual people, we may want to find their sales
# rank within each state

rng = np.random.default_rng(10 * 8)
df = pd.DataFrame({"id":[x for x in range(10)],
                    "state":["MI", "WI", "OH", "OH", "MI", "MI", "WI", "WI",
                                "OH", "MI"],
                    "sales":rng.integers(low=10, high=100, size=10)})
df

# ## groupby -> rank
# - Need to chain `groupby()` and `rank()`
# - Assign results to new ranking column
# - Now we can easily see the best sales person in each state and can do
# additional filtering on this column
# - Especially useful for time-based ranking
#     - E.g. Find last 5 appointments for each patient

df["sales_rank"] = (df.groupby("state")["sales"]
                        .rank(method="first", ascending=False))
df

top_2 = (df.query('sales_rank < 3')
            .sort_values(by=["state", "sales_rank"]))
top_2


# # name: Siwei Tang Email: tangsw@umich.edu
# # Q0
# ## Time series/ data functionality
#
# The Python world has a number of available representation of dates, times, deltas, and timespans. Whiles the times series tools provided by Pandas tend to be the most useful for data science applications, it's helpful to see their relationsip to other packages used in Python.
#
# ## Native Python dates and times: `datetime` and `dateutil`
#
# Pythonn's baseic objects for working with dates and times reside in the built-in `dateime` module. Along with the third-party `dateutil` module, you can use it to quickly perform a host of useful functionalities on dates and time. 

# - build a date using the `datetime` type

from datetime import datetime
datetime(year = 2021, month=10, day=20)

# - using dateutil module to parse dates from a variety of strng formats

from dateutil import parser
date = parser.parse("20th of October, 2021")
date 

# - Once you have a `datetime` object, you can do things like printing the day of the week:

date.strftime('%A')

# In the final line, `%A` is part of the [strfyime section](https://docs.python.org/3/library/datetime.html) od Python's [datetime documentation]()

# ## Typed arrays of times: Numpy's `datatime64`
# - The `datatime64` dtype encoded dates as 64-bit inegers, and thus allows arrays of dates to be represented very compactly. The `datatime64` requires a very specific input format:

date =np.array('2021-10-20', dtype=np.datetime64)
date

# - Once we have this date formated, however, we can quickly do vectorized operations on it

date + np.arange(12)

# - One detail of the `datetime64` and `timedelta64` object is that they are build on a fundamental time unit. Because the `datetime64` object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times this fundamental unit. In other words, `datetime64` imposes a trade-off between **time resolution** and **maximum time span**.

# ## Dates and times in pandas: best of both worlds
# Pandas builds upon all the tools just discussed to provide a `Timestamp` object, which combines the ease-of-use of `datetime` and `dateutil` with the efficient storage and vectorized interface of `numpy.datetime64`. From a group of these `Timestamp` objects, Pandas can construct a `DatetimeIndex` that can be used to index data in a `Series` or `DataFrame`.

date = pd.to_datetime('20th of October, 2021')
date

date.strftime('%A')

# - we can do Numpy-style vectorized operations directly on this same object:

date + pd.to_timedelta(np.arange(12),'D')

# ## Pandas Time Series Data Structures
# - for time stamps, Pandas provides the `Timestamp` type. As mentioned before, it is essentially a replacement for Python's native `datetime`, but is based on the more efficient `numpy.datetime64` date type. The associated Index structure is `DatetimeIndex`. 
# - for time Periods, Pandas provides the `Period` type. This encodes a fixed-frequency interval based on `numpy.datetime64`. The associated index structure is `PeriodIndex`.
# - For time deltas or durations, Pandas provides the `Timedelta` type. `Timedelta` is a more efficient replacement for Python's native `datetime.timedelta` type, and is based on `numpy.timedelta64`. The assocaited index structure is `TimedeltaIndex`.
#
# Passing a single date to `pd.to_datetime()` yields a `Timestamp`; passing a series of dates by default yields a `DatetimeIndex`:

dates = pd.to_datetime([datetime(2021,10,20),
                        '21st of October, 2021',
                        '2021-Oct-22',
                       '10-23-2021',
                       '20211024'])
dates

# - Any `DatetimeIndex` can be converted to a `PeriodIndex` with the `to_period()` function with the addition of a frequency code; here we use `'D'` to indicate daily frequency.

dates.to_period('D')

# - A `TimedeltaIndex` is created, for example, when a date is subtracted from another:

dates - dates[0]

# ## Regular Sequences: `pd.date_range()`

# - `pd.date_range()` for timestamsps, `pd.period_range()` for periods, and `pd.timedelta_range()` for time deltas. This is similar to Python's `range()` or `np.arange()`.

pd.date_range('2021-09-11','2021-10-21')

# - Alternatively, the date range can be specified not with a start and end point, but with a startpoint and a number of periods
# - The spacing can be modified by altering the `freq` argument, which defaults to `D`.

print(pd.date_range('2021-09-11',periods=10))
print(pd.date_range('2021-09-11', periods = 10, freq = 'H'))

# - To create regular sequencs of `Period` or `Timedelta` values, the very similar `pd.period_range()` and `pd.timedelta_range()` functions are useful. Here are some monthly periods:

pd.period_range('2021-09',periods = 10, freq='M')

# - A sequence of durations increasing by an hour:

pd.timedelta_range(0,periods=30, freq='H')


# Zehua Wang wangzeh@umich.edu

# ## Imports

import pandas as pd

# ## Question 0 - Topics in Pandas [25 points]

# ## Data Cleaning

# Create sample data
df = pd.DataFrame(
    {
        'col1': range(5),
        'col2': [6, 7, 8, 9, np.nan],
        'col3': [("red", "black")[i % 2] for i in range(5)],
        'col4': [("x", "y", "z")[i % 3] for i in range(5)],
        'col5': ["x", "y", "y", "x", "y"]
    }
)
df

# ### Duplicated Data
# - Find all values without duplication
# - Check if there is duplication using length comparison
# - return true if duplication exists

df['col3'].unique()
len(df['col3'].unique()) < len(df['col3'])

# ### Duplicated Data
# - Record duplication
# - subset: columns that need to remove duplication. Using all columns
#   if subset is None.
# - keep: Determine which duplicates to keep (if any), 'first' is default
#     - 'first': drop duplications except the first one
#     - 'last': drop duplications except the last one
#     - False: drop all duplications
# - inplace: return a copy (False, default) or drop duplicate (True)
# - ignore_index: return series label 0, 1, ..., n-1 if True, default is False

df.drop_duplicates()
df.drop_duplicates(subset=['col3'], keep='first', inplace=False)
df.drop_duplicates(subset=['col4', 'col5'], keep='last')

# ### Missing Data
# - Check if there is missing value
# - Delete missing value: pd.dropna
#     - axis: 0, delete by row; 1, drop by column
#     - how: any, delete if missing value exist; all, delete if 
#         all are missing values
#     - inplace: return a copy (False, default) or drop duplicate (True)    

df.isnull().any() # pd.notnull for selecting non-missing value
df.dropna(axis=0, how='any')

# ### Missing Data
# - Replcae missing value: pd.fillna
#     - value: the value filled up for missing value
#     - method: how to fill up the missing value
#         - 'backfill'/'bfill': using next valid observation
#         - 'pad'/'ffill': using previous valid observation
#         - None is by default
# - Generally, we could fill up the missing value with mean or median
#     for numeric data, and mode in categorical data.

df.fillna(method='ffill')
df.fillna(value=np.median(df[df['col2'].notnull()]['col2']))

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Filling in Missing Data](#Filling-in-Missing-Data) 
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide. 

# ## Filling in Missing Data
#
#
# *Xinhe Wang*
#
# xinhew@umich.edu

# ## Fill in Missing Data
#
# - I will introduce some ways of using ```fillna()``` to fill in missing 
# data (```NaN``` values) in a DataFrame.
# - One of the most easiest ways is to drop the rows with missing values.
# - However, data is generally expensive and we do not want to lose all 
# the other columns of the row with missing data.
# - There are many ways to fill in the missing values:
#     - Treat the ```NaN``` value as a feature -> fill in with 0;
#     - Use statistics -> fill in with column mean/median/percentile/a
#     random value;
#     - Use the "neighbors" -> fill in with the last or next values;
#     - Prediction methods -> use regression/machine learning models to 
#     predict the missing value.

# ## Example Data
# - Here we generate a small example dataset with missing values.
#
# - Notice that if we want to indicate if the value in column "b" is larger
# than 0 in column "f", but for the missiing value in column "b", 
# ```df['b'] > 0``` returns ```False``` instead of a ```NaN``` value.
# Therefore, ```NaN``` values need to be delt with before further steps.

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 4),
                  columns=['a', 'b', 'c', 'd'])
df.iloc[2, 1] = np.nan
df.iloc[3:5, 0] = np.nan
df['e'] = [0, np.nan, 0, 0, 0]
df['f'] = df['b']  > 0
df

# ## Fill in with a scalar value
# - We can fill in ```NaN``` values with a designated value using 
# ```fillna()```.

df['e'].fillna(0)

df['e'].fillna("missing")

# ## Fill in with statistics (median, mean, ...)
# - One of the most commonly used techniques is to fill in missing values
# with column median or mean.
# - We show an instance of filling in missing values in column "b" with 
# column mean.

df['b'].fillna(df.mean()['b'])

# ## Fill in with forward or backward values
# - We can fill in with the missing values using its "neighber" using 
# ```fillna()```.
# - Can be used if the data is a time series.
# - When the ```method``` argument of ```fillna()``` is set as ```pad``` 
# or ```ffill```, values are filled forward; when ```method``` is set as
# ```bfill```or ```backfill```, values are filled backward.
# - The ```limit``` argument of ```fillna()``` sets the limit of number 
# of rows it is allowed to fill.

df['a'].fillna(method='pad', limit=1)

df['a'].fillna(method='bfill', limit=1)

# <p>This is a short tutorial about neat pandas idioms. <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#idioms">idioms</a> .
# From Xiatian Chen email:simoncxt@umich.edu</p>
# <h1>Idioms</h1>
# <h2>If-then and splitting:</h2>
# <pre><code>    -Clear idioms allow the code to be more readable and efficient  
#     -Always need to construct data under specific conditions, here are some examples.
# </code></pre>
# <p><code>df = pd.DataFrame(
#     {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
# )
# df.loc[df.AAA &gt;= 5, "BBB"] = -1</code>  </p>
# <pre><code>    -Can also apply if-then to multiple columns
# </code></pre>
# <p><code>df.loc[df.AAA &gt;= 5, ["BBB", "CCC"]] = 555</code>  </p>
# <pre><code>    -Can use numpy where() to apply if-then-else
# </code></pre>
# <p><code>df["logic"] = np.where(df["AAA"] &gt; 5, "high", "low")</code>  </p>
# <pre><code>    -Split the frame under condition
# </code></pre>
# <p><code>df[df.AAA &lt;= 5]
# df[df.AAA &gt; 5]</code> </p>
# <h2>Building criteria:</h2>
# <pre><code>    -When there is only 1-2 criterias, can be directly contained in df.loc  
#     -Can return a series or just modify the dataframe
# </code></pre>
# <p><code>df.loc[(df["BBB"] &lt; 25) &amp; (df["CCC"] &gt;= -40), "AAA"]
# df.loc[(df["BBB"] &gt; 25) | (df["CCC"] &gt;= 75), "AAA"] = 0.1</code>   </p>
# <pre><code>    -When there is a list of criteria, it can be done with a list of dynamically built criteria
# </code></pre>
# <p><code>Crit1 = df.AAA &lt;= 5.5
# Crit2 = df.BBB == 10.0
# Crit3 = df.CCC &gt; -40.0
# CritList = [Crit1, Crit2, Crit3]
# AllCrit = functools.reduce(lambda x, y: x &amp; y, CritList)
# df[AllCrit]</code> </p>

# In[ ]:


# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ### Author: Houyu Jiang
# ### Email: houyuj@umich.edu

# + [Topic: pd.diff()](#Topic:-pd.diff()) 
# + [Direction of the difference](#Direction-of-the-difference)
# + [Distance of difference](#Distance-of-difference)

# ## Topic: pd.diff()
#
# - ```pd.diff()``` is a pandas method that we could use to
# compute the difference between rows or columns in DataFrame.
# - We could import it through ```import pandas as pd```.
# - Suppose ```df``` is a pandas DataFrame, we could use 
# ```diff``` method to compute.

df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                   'b': [1, 1, 2, 3, 5, 8],
                   'c': [1, 4, 9, 16, 25, 36]})
df.diff(axis=0)

# ## Direction of the difference
# - ```pd.diff()``` by default would calculate the 
# difference between different rows.
# - We could let it compute the difference between 
# previous columns by setting ```axis=1```

df.diff(axis=1)

# ## Distance of difference
# - ```pd.diff()``` by default would calculate the difference
# between this row/column and previous row/column
# - We could let it compute the difference between this row/column
# and the previous n row/column by setting ```periods=n```

df.diff(periods=3)


# In[ ]:


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Missing Data](#Missing-Data)

# ## Missing Data
# I will be looking at how pandas dataframes handle missing values.
# **Stefan Greenberg**
#
# sfgreen@umich.edu


# ## Imports
import numpy as np
import pandas as pd
# ## Detecting missing data
# - missing data includes `NaN`, `None`, and `NaT`
#     - can change settings so that `inf` and -`inf` count as missing
# - `.isna()` returns True wherever there is a missing value
# - `.notna()` returns True wherever there is not a missing value

# +
df = pd.DataFrame([[0.0, np.NaN, np.NaN, 3.0, 4.0, 5.0],
                   [0.0, 1.0, 4.0, np.NaN, 16.0, 25.0]], 
                 index=['n', 'n^2'])

df.append(df.isna())
# -

# ## Filling missing data
#
# - pandas makes it easy to replace missing values intelligently
# - the `.fillna()` method replaces all missing values with a given value
# - the `.interpolate()` method will use neighboring values to fill in gaps
# in data

# +
df_zeros = df.fillna(0)
df_interp = df.copy()

df_interp.loc['n'] = df_interp.loc['n']                     .interpolate(method='linear')
df_interp.interpolate(method='quadratic', axis=1, inplace=True)

df_zeros
#df_interp
# -

# ## Remove missing data with `.dropna()`
#
# - `.dropna()` will remove rows or columns that have missing values
# - set `axis` to determine whether to drop rows or columns
# - drop a row or column if it has any missing values or only if it has 
# entirely missing values by setting `how` to either *'any'* or *'all'*
# - set a minimum number of non-missing values required to drop row/column
# by setting `thresh`
# - specify what labels along other aixs to look at using `subset` i.e. 
# only drop a row if there is a missing value in a subset of the columns 
# or vise versa

# +
drop_cols   = df.dropna(axis=1)
drop_all    = df.dropna(how='all')
drop_thresh = df.dropna(thresh=5)
drop_subset = df.dropna(subset=[0, 1, 5])

print(df, '\n\n', 
      drop_cols.shape, drop_all.shape, drop_thresh.shape, drop_subset.shape)
# -
# ## Math operations with missing data
# - cumulative methods - `.cumsum()` and `.cumprod()` - by default will skip 
# over missing values
# - `.sum()` and `.prod()` treat missing values as identities
#     - `.sum()` treats missing values as zero
#     - `.prod()` treats missing values as one
#


# +
sumprod = df.append(
          df.sum()
            .to_frame()
            .transpose()
            .rename(index={0:'sum'}))

sumprod.append(
        df.prod()
          .to_frame()
          .transpose()
          .rename(index={0:'prod'}))