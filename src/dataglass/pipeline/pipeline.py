"""
This module defines the base classes for building a modular and extensible data processing pipeline.

It includes:
- `PipelineStep`: An abstract base class that defines the interface for each pipeline step.
- `DataPipeline`: A container that sequentially applies a list of `PipelineStep` instances to a pandas DataFrame.

Classes
-------
PipelineStep : ABC
    Abstract base class for any step that can be used in a data pipeline. 
    Requires implementing the `apply` method.

DataPipeline
    A class that executes a sequence of pipeline steps on a given DataFrame.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class _PipelineStep(ABC):
    """
    Abstract base class for all data preprocessing pipeline steps.

    Any custom pipeline step should inherit from this class and implement the `apply` method.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame
        Applies the transformation to the input DataFrame and returns the modified DataFrame.
    """
    @abstractmethod
    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        pass


class DataPipeline():
    """
    A class to manage and execute a sequence of data preprocessing steps.

    Parameters
    ----------
    steps : List[PipelineStep]
        A list of preprocessing steps that inherit from the PipelineStep base class.

    Methods
    -------
    apply(data: pd.DataFrame) -> pd.DataFrame
        Applies all the steps sequentially to the input DataFrame and returns the transformed data.
    """
    def __init__(self, steps : List[_PipelineStep]):
        self.steps = steps

    def apply(self, data : pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            data = step.apply(data)
        
        return data