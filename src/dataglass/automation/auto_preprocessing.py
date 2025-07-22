from numpy import dtype
from ..preprocessing import *
import pandas as pd
from typing import List

def has_datetime(data: pd.DataFrame) -> List:
    # Datatype heuristic inference
    data = convert_datatype_auto(data)

    datetime_columns = []
    dtypes = data.dtypes
    for dt in dtypes:
        if pd.api.types.is_datetime64_any_dtype(dt):
            datetime_columns.append(dtypes)