import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TARGET = "income"
CAT = [
    "workclass","education","marital-status","occupation","relationship","race","sex","native-country",
]
NUM = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = (out[TARGET].astype(str).str.strip() == ">50K").astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])