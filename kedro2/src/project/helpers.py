import pandas
import sklearn.linear_model
import sklearn.metrics


def drop_columns_with_missing_values(data: pandas.DataFrame, threshold: float) -> pandas.DataFrame:
    column_mask = (data.isna().sum(axis="index") / data.shape[0]) > threshold
    data = data.drop(columns=data.columns[column_mask])
    if data.shape[1] == 0:
        raise ValueError("No columns left after dropping")
    return data


def impute_columns_with_mean(data: pandas.DataFrame) -> pandas.DataFrame:
    data = data.fillna(data.mean(numeric_only=True))
    return data


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def fit_linear_model(
    X: pandas.DataFrame, y: pandas.DataFrame
) -> tuple[sklearn.linear_model.LinearRegression, float]:
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    r2 = sklearn.metrics.r2_score(y, model.predict(X))
    r2_adj = adjusted_r2_score(r2=r2, n=X.shape[0], p=X.shape[1])  # type: ignore
    return (model, r2_adj)
