from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import mlflow
from prefect import task, flow

@task
def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data


@task
def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())


@task
def get_scaler(data: pd.DataFrame) -> Any:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


@task
def rescale_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    data_rescaled = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return data_rescaled


@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}