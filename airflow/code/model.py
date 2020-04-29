# %%
import argparse
from toolz import excepts
import pandas as pd
import seaborn as sns
import numpy as np
import os.path as op
#import os
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import norm, uniform
import pickle

# %%
INPUT = "env_raw/data"
OUTPUT = "env_raw/data"
INTERACTIVE = excepts(
    Exception,
    lambda: get_ipython() == get_ipython(), lambda _: False
)()
RANDOM_STATE = 42
np.random.RandomState(RANDOM_STATE)

#%%
if not INTERACTIVE:
    parser = argparse.ArgumentParser(
        description='Prepares the dataset for the classification task'
    )
    parser.add_argument(
        '-i',
        "--input",
        help='The input where the csvs are',
        default=INPUT
    )
    parser.add_argument(
        '-o',
        "--output",
        help='The output where the model will be saved',
        default=OUTPUT
    )
    args = parser.parse_args()
    INPUT = args.input
    OUTPUT = args.output

# %%
df = pd.read_csv(op.join(INPUT, "dataset.csv"))

print(f"Got {len(df)} entries")


#%%
COL_NAMES = [
    x for x in df.columns if
    x in ["home__val", "away__val"] or
    "__avg_" in x or
    "delta_minus" in x
]
TARGET = "delta"

#%%
print(f"Dataframe sample:\n{df.head()}")
# %%
df.loc[df[TARGET] > 0, TARGET] = 1
df.loc[df[TARGET] < 0, TARGET] = -1
#%%
X = df[COL_NAMES]
y = df[TARGET]


#%%
X_tr, y_tr = X[:-200], y[:-200]
X_te, y_te = X[-200:], y[-200:]

splitter = TimeSeriesSplit(n_splits=3)

param_grid = {
    "min_samples_split": uniform(loc=2, scale=8),
    "min_samples_leaf": uniform(loc=1, scale=5),
    "max_depth":uniform(loc=5, scale=40)
}

p_sampler = ParameterSampler(
    param_grid,
    n_iter=1000,
    random_state=RANDOM_STATE
)
#%%
data = []
for p in p_sampler:
    params = {k:int(v) for k, v in p.items()}
    res = []
    for tr_ind, val_ind in splitter.split(X_tr):
        X_tr_fold, y_tr_fold = X_tr.iloc[tr_ind], y_tr.iloc[tr_ind]
        X_te_fold, y_te_fold = X_tr.iloc[val_ind], y_tr.iloc[val_ind]
        dt = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            **params
        )
        dt.fit(X_tr_fold, y_tr_fold)
        res.append(dt.score(X=X_te_fold, y=y_te_fold))
    #print(f"Got {np.mean(res)} for params {params}")
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        **params
    )
    dt.fit(X_tr, y_tr)
    data.append(
        {
            "score":np.mean(res),
            **params,
            "model":dt
        }
    )
# %%
df_search = pd.DataFrame(data).drop_duplicates(subset=[*param_grid,])
df_search.sort_values(by="score", inplace=True, ascending=False)
# %%
print(f"A brief of the results:\n{df_search.head()}")
# %%
print(f"Best model score on test set {df_search.loc[0].model.score(X_te, y_te)}")

output_file_path = op.join(OUTPUT, "model.pkl")
os.makedirs(OUTPUT, exist_ok=True)
with open(output_file_path, "wb") as f:
    pickle.dump(df_search.loc[0].model, f)
# %%
