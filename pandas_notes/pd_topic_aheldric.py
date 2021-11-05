from collections import defaultdict
from timeit import Timer
from typing import final
import numpy as np
import pandas as pd
from scipy.stats import t, beta
from scipy import stats
from IPython.core.display import display, Markdown, HTML
import pickle
import matplotlib.pyplot as plt

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