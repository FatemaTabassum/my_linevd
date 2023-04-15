import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sastvd as svd
import sastvd.helpers.git as svdg

from IPython.display import display

import sastvd as svd
#svd.get_dir
from multiprocessing import Pool
from tqdm import tqdm



def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)

    


def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    print("huifhuifohuofhweofh hfgfuhfuifgh hfuhyfuhyf")
    
    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
            
        
    return processed




def bigvul(minimal=True, sample=False, return_raw=False, splits="default"):
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")

    if minimal:
        try:
            df = pd.read_parquet(savedir/ f"minimal_bigvul_{sample}.pq", engine="fastparquet").dropna()
            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            default_splits = external_dir() / "bigvul_rand_splits.csv"

            if os.path.exists(default_splits):
                #I renamed splits as splits_df for better understanding
                splits_df = pd.read_csv(default_splits)
                # splits_df = splits_df.head(10) ##Liza taking only the first 10 rows
                splits_dic    = splits_df.set_index("id").to_dict()["label"]
                #print(splits_dic)
                #print(df.head(15))
                #Liza with this statement they are replacing the old id, label columns with the given id, column from default_splits. 
                #Replacing df[label] with the value 'id' is mapping to in splits_dic
                df["label"] = df.id.map(splits_dic)
                #print(df.head(35))

            if "crossproject" in splits_dic:
                print("crossproject")

            return df

        except Exception as E:
            print(E)
            pass
    
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    print(filename)
    df = pd.read_csv(svd.external_dir() / filename)
    #Renaming the unnamed column to id
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"
    display(df.head(10))
    
    #Remove Comments
    df["func_before"] = dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = dfmp(df, remove_comments, "func_after", cs=500)
    
    # Return raw for testing
    if return_raw:
        return df
    
    # save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs = 300)
    
    # Assign info and save
    df["info"] = dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    return df



df = bigvul(minimal=False)

