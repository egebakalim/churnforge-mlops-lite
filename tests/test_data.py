import pandas as pd
from src.data import basic_normalize, split_xy, TARGET_COL


def test_basic_normalize_churn_to_int():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "Churn": ["Yes", "No", "Yes"],
        }
    )
    out = basic_normalize(df)
    assert out[TARGET_COL].dtype.kind in ("i", "u")
    assert set(out[TARGET_COL].unique().tolist()) <= {0, 1}


def test_split_xy_works():
    df = pd.DataFrame({"x": [1, 2, 3], "Churn": [0, 1, 0]})
    X, y = split_xy(df)
    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert len(X) == len(y)
