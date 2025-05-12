import numpy as np
import pandas as pd


def datetime_to_time(s: pd.Series) -> pd.Series:
    """Convert a datetime col/series to time = %H:%M:%S"""
    try:
        s = pd.to_datetime(s, errors='coerce')  # Convert to datetime, invalid values become NaT
        time_series = s.dt.time  # get time object
        return time_series
    except Exception as e:
        print(f"Error in datetime conversion: {e}")
        return pd.Series()  # Return an empty series

