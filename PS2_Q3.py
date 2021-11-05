from collections import defaultdict
from timeit import Timer
import time
import numpy as np
import pandas as pd
from scipy.stats import t, beta
from scipy import stats
from IPython.core.display import display, Markdown
import pickle
import matplotlib.pyplot as plt

def oral_col_name(col_name):
    """
    Updates column names for NHANES oral health data

    Parameters
    ----------
    col_name : string
        current column name

    Returns
    -------
    string - new column name

    """

    new_name = col_name 

    ##for tooth count / coronal columns, extract the number and append to name
    if "CTC" in col_name:
        new_name = "coronal_caries_" + col_name[3:5]
    elif "TC" in col_name:
        new_name = "tooth_count_" + col_name[3:5]
    elif col_name == "SEQN":
        new_name = "id"
    elif col_name == "OHDDESTS":
        new_name = "ohx_status"

    return new_name

def get_nhanes_data(file_prefix):
    """
    Collects NHANES data from XPT files

    Parameters
    ----------
    file_prefix : string
        prefix for file name to load

    Returns
    -------
    pd.DataFrame - dataframe with 4 cohorts combined

    """

    cohort_year =  {"G":2011, "H":2013, "I":2015, "J":2017}
    
    cohorts = []
    for yr in cohort_year:
        data = pd.read_sas(
                file_prefix + "_" + yr + ".XPT",
                format="xport",
                encoding="utf-8")
        data['cohort_year'] = cohort_year[yr]
        cohorts.append(data)

    return pd.concat(cohorts)

def update_category_col(col_series, cat_codes=[-1], cat_names=["Missing"]):
    """
    Updates column to be categorical. Fills in missing data w/ -1 and encodes
    as "Missing". We need to manually set the -1 category in case the data does
    not include any missing values

    Parameters
    ----------
    col_series : pd.Series
        series containing column data
    cat_codes : list
        numeric codes that represent current data. Should include -1
        for missing data
    cat_names : list
        names for each category to be renamed as

    Returns
    -------
    pd.Series - data for new column

    """
    
    new_col = (col_series.fillna(-1)
                .astype('category')
                .cat.set_categories(cat_codes)
                .cat.rename_categories(cat_names))

    return new_col

def clean_oral_hx_data(df):
    """
    Performs data cleaning on oral health data

    Parameters
    ----------
    df : pd.DataFrame
        oral health dataframe

    Returns
    -------
    pd.DataFrame - dataframe with columns names and dtypes updated

    """

    ##only select columns we want
    tooth_col_start = 4
    tooth_col_end = 64
    oral_cols = (["SEQN", "OHDDESTS"] + 
        list(df.columns)[tooth_col_start:tooth_col_end] + ['cohort_year'])
    
    ##rename columns and change data types
    df = (df[oral_cols]
            .rename(columns=oral_col_name)
            .astype({"id":int}))  

    #df['status'] = update_category_col(
    #                df['status'],
    #                [-1, 1, 2, 3],
    #                ["Missing", "Complete", "Partial", "Not Done"])
    df['ohx_status'] = df['ohx_status'].astype("category")
    
    ##for tooth counts we'll use description as category names
    ##but for coronals we'll use existing chars
    tooth_cnt_cat_codes = [-1, 1, 2, 3, 4, 5, 9]
    tooth_cnt_cat_names = ["Missing", "Primary tooth (deciduous) present",
                            "Permanent tooth present", "Dental implant",
                            "Tooth not present",
                            "Permanent dental root fragment present",
                            "Could not assess"]
    
    for col in df.columns:
        if "tooth" in col:
            df[col] = update_category_col(
                        df[col],
                        tooth_cnt_cat_codes,
                        tooth_cnt_cat_names)
        elif "coronal" in col:
            df[col] = df[col].astype("category")
    
    return df

def clean_demo_data(df):
    """
    Performs data cleaning on demographics data

    Parameters
    ----------
    df : pd.DataFrame
        demographics dataframe

    Returns
    -------
    pd.DataFrame - dataframe with columns names and dtypes updated

    """

    ##rename columns and change data types
    demo_cols = {"SEQN":"id", "RIDAGEYR":"age", "RIDRETH3":"race_ethnicity",
                    "DMDEDUC2":"education", "DMDMARTL":"marital_status",
                    "RIDSTATR":"exam_status", "SDMVPSU":"variance_psu",
                    "SDMVSTRA":"variance_stratum",
                    "WTMEC2YR":"mec_exam_weight",
                    "WTINT2YR":"interview_weight", "RIAGENDR":"gender"}

    df = (df[list(demo_cols.keys()) + ['cohort_year']]
            .rename(columns=demo_cols)
            .astype({"id":int, "age":np.int8, "variance_psu":int,
                        "variance_stratum":int}))

    ##update categorical columns
    #stat_cat_names = ["Missing", "Interviewed only",
    #                    "Both interviewed and MEC examined"]
    #df['status'] = update_category_col(
    #                    df['status'],
    #                    [-1, 1, 2],
    #                    stat_cat_names)
    df["exam_status"] = df["exam_status"].astype("category")

    re_cat_names = ["Missing", "Mexican American", "Other Hispanic",
                    "Non-Hispanic White", "Non-Hispanic Black",
                    "Non-Hispanic Asian", "Other Race - Including Multi-Racia"]
    df['race_ethnicity'] = update_category_col(
                                df['race_ethnicity'],
                                [-1, 1, 2, 3, 4, 6, 7],
                                re_cat_names)

    edu_cat_names = ["Missing", "Less than 9th grade", "9-11th grade", 
                        "High school graduate/GED",
                        "Some college or AA degree",
                        "College graduate or above", "Refused", "Don't Know"]
    df['education'] = update_category_col(
                            df['education'],
                            [-1, 1, 2, 3, 4, 5, 7, 9],
                            edu_cat_names)
    
    marry_cat_names = ["Missing", "Married", "Widowed", "Divorced",
                        "Separated", "Never married", "Living with partner",
                        "Refused", "Don't Know"]
    df['marital_status'] = update_category_col(
                                df['marital_status'],
                                [-1, 1, 2, 3, 4, 5, 6, 77, 99],
                                marry_cat_names)

    df["gender"] = update_category_col(
                                df['gender'],
                                [-1, 1, 2],
                                ["Missing", "Male", "Female"])

    return df

##load data
demo_data = get_nhanes_data("DEMO")
oral_data = get_nhanes_data("OHXDEN")

##clean data
demo_data = clean_demo_data(demo_data)
oral_data = clean_oral_hx_data(oral_data)

##save data
demo_data.to_pickle("nhanes_demo.pickle")
oral_data.to_pickle("nhanes_ohx.pickle")


# Determine the total number of cases in each dataset.

num_cases_demo = demo_data.shape[0]
num_cases_oral = oral_data.shape[0]
print(f'Number of cases in demographics data: {num_cases_demo}')
print(f'Number of cases in oral health data: {num_cases_oral}')