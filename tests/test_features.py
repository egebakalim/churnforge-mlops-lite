import pandas as pd
from src.features import build_preprocessor


def test_preprocessor_builds_and_transforms():
    X = pd.DataFrame(
        {
            "age": [20, 30, None],
            "plan": ["A", "B", "A"],
        }
    )
    fb = build_preprocessor(X)
    Xt = fb.preprocessor.fit_transform(X)

    # Expect: numeric 1 col + categorical OHE (2 categories => 2 cols)
    assert Xt.shape[0] == 3
    assert Xt.shape[1] >= 3
