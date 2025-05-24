import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__doc__ = """
This module contains functions for preprocessing data.
It includes functions to load data, clean it, and save the cleaned data to a specified directory.
"""


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by removing rows with missing values and duplicates.
    Drop the unnecessary columns 'salary' and 'salary_currency'.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be cleaned.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with missing values and duplicates removed.
    """
    # Remove rows with missing values
    data = data.dropna()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Drop unnecessary columns
    data = data.drop(columns=["salary", "salary_currency"], errors="ignore")

    return data


def process_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process categorical features in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with categorical features to be processed.

    Returns
    -------
    pd.DataFrame
        The DataFrame with processed categorical features.
    """

    data["company_size"] = data["company_size"].map(
        {
            "S": 1,
            "M": 2,
            "L": 3,
        }
    )
    data["experience_level"] = data["experience_level"].map(
        {
            "EN": 1,
            "MI": 2,
            "SE": 3,
            "EX": 4,
        }
    )
    data = pd.get_dummies(
        data,
        columns=["employment_type"],
        prefix="employment_type",
        drop_first=True,
        dtype=int,
    )

    return data


def preprocess_country(data: pd.DataFrame, gdp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the column that use country code by replacing it with the GDP value.
    The effected column are 'company_location' and `employee_residence`.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing country codes.
    gdp_data : pd.DataFrame
        The DataFrame containing GDP values indexed by country codes.

    Returns
    -------
    pd.DataFrame
        The DataFrame with country codes replaced by GDP values.
    """

    gdp_data = gdp_data.set_index("alpha_2")

    # Replace country codes with GDP values
    data.loc[:, "company_location"] = data.apply(
        lambda x: gdp_data.loc[x["company_location"], str(x["work_year"])],
        axis=1,
    )
    data.loc[:, "employee_residence"] = data.apply(
        lambda x: gdp_data.loc[x["employee_residence"], str(x["work_year"])],
        axis=1,
    )

    return data


def preprocess_numbers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the numeric features in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with numeric features to be processed.

    Returns
    -------
    pd.DataFrame
        The DataFrame with processed numeric features.
    """

    # Convert 'work_year' to int
    data["work_year"] = data["work_year"].astype(int)

    # Winsorize the 'salary_in_usd' column to remove outliers
    data["salary_in_usd"] = winsorize(data["salary_in_usd"], limits=[0.01, 0.01])

    return data


def preprocess_title(data: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    """
    Preprocess the 'job_title' column in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with a 'job_title' column to be processed.

    Returns
    -------
    pd.DataFrame
        The DataFrame with processed 'job_title' column.
    """

    # Vectorize the 'job_title' column
    job_title_vector = pd.DataFrame(
        vectorizer.transform(data["job_title"]).toarray(),
        columns=vectorizer.get_feature_names_out(),
    ).add_prefix("job_title_", axis=1)

    data = data.drop(columns=["job_title"], errors="ignore")
    data = pd.concat([data, job_title_vector], axis=1)

    return data


def preprocess_data(data: pd.DataFrame, gdp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by cleaning, processing categorical features,
    replacing country codes with GDP values, processing numeric features, and
    vectorizing the 'job_title' column.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be preprocessed.
    gdp_data : pd.DataFrame
        The DataFrame containing GDP values indexed by country codes.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """

    data = clean_data(data)
    data = process_categoricals(data)
    data = preprocess_country(data, gdp_data)
    data = preprocess_numbers(data)

    # Initialize CountVectorizer for 'job_title'
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["job_title"])

    data = preprocess_title(data, vectorizer)

    return data
