import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TARGET = "income"
CAT = [
    "workclass","education","marital_status","occupation","relationship","race","sex","native_country",
]
NUM = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # normalize column names
    out.rename(columns=lambda col: col.replace('-','_'), inplace=True) 
    # Remove outlier values
    out[TARGET] = (out[TARGET].astype(str).str.strip() == ">50K").astype(int) 
    return out


def build_preprocessor() -> ColumnTransformer:
    # Column encoding
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])