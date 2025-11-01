
import pandas as pd

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Empty CSV")
    return df

def pick_default_target(df: pd.DataFrame) -> str:
    nums = df.select_dtypes("number").columns
    if len(nums) == 0:
        raise ValueError("No numeric column to use as target")
    return nums[0]
