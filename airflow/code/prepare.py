#%%
import pandas as pd
import numpy as np
from functools import partial
import os.path as op
import argparse
from toolz import excepts
# %%
def prev_val(prev, group):
    return pd.Series(
        group.values[prev:].tolist() + [0]*prev,
        index=group.index
    )

def avg(window, group):

    convolution = np.convolve(
        group.values[1:],
        np.ones((window,))/window, mode="valid"
    ).tolist()
    return pd.Series(
        convolution + [0]*(len(group) - len(convolution)),
        index=group.index
    )

#%%
INPUT = "env_raw/data"
OUTPUT = "env_raw/data"
INTERACTIVE = excepts(
    Exception,
    lambda: get_ipython() == get_ipython(), lambda _: False
)()
# %%
if not INTERACTIVE:
    parser = argparse.ArgumentParser(
        description='Prepares the dataset for the classification task'
    )
    parser.add_argument(
        '-o',
        "--output",
        help='The output where the result is to be saved',
        default=OUTPUT
    )
    parser.add_argument(
        '-i',
        "--input",
        help='The input where the csvs are',
        default=INPUT
    )
    args = parser.parse_args()
    INPUT = args.input
    OUTPUT = args.output
# %%
df_v = pd.read_csv(op.join(INPUT, "values.csv"))
df_g = pd.read_csv(op.join(INPUT, "games.csv"))
# %%
df_g["delta"] = df_g.goals_home - df_g.goals_away
# %%
df_g["date"] = pd.to_datetime(df_g.date)
df_g["year"] = df_g.date.dt.year
# %%
df_g = pd.merge(
    df_g,
    df_v.add_prefix("home__"),
    how="left",
    left_on=["year", "home"],
    right_on=["home__year", "home__name"]
).drop(columns=["home__name", "home__year"])
# %%
df_g = pd.merge(
    df_g,
    df_v.add_prefix("away__"),
    how="left",
    left_on=["year", "away"],
    right_on=["away__year", "away__name"]
).drop(columns=["away__name", "away__year"])

# %%
df_home = df_g[
    [
        "date",
        "home",
        "delta",
        "goals_home",
        "goals_away",
        "season"
    ]
].copy()
df_home["team"] = df_home.home
df_home["scored"] = df_home.goals_home
df_home["suffered"] = df_home.goals_away
df_home.drop(columns=["goals_home", "goals_away"], inplace=True)
df_home["home"] = True

df_away = df_g[
    [
        "date",
        "away",
        "delta",
        "goals_home",
        "goals_away",
        "season"
    ]
].copy()
df_away["team"] = df_away.away
df_away["scored"] = df_away.goals_away
df_away["suffered"] = df_away.goals_home
df_away["delta"] = df_away.delta*-1
df_away.drop(columns=["away", "goals_home", "goals_away"], inplace=True)
df_away["home"] = False

df_aux = pd.concat([df_home, df_away])
df_aux.sort_values(by="date", ascending=False, inplace=True)

# %%
for i in range(1, 5):
    df_aux[f"delta_minus_{i}"] = df_aux.groupby(
        ["team", "season"]
    ).delta.transform(partial(prev_val, i))

# %%
for i in [3, 5]:
    df_aux[f"avg_scored_{i}"] = df_aux.groupby(
        ["team", "season"]
    ).scored.transform(partial(avg, i))
    df_aux[f"avg_suffered_{i}"] = df_aux.groupby(
        "team"
    ).suffered.transform(partial(avg, i))
# %%
df_g = pd.merge(
    df_g,
    df_aux.add_prefix("home__"),
    how="left",
    left_on=["date", "home"],
    right_on=["home__date", "home__team"]
).drop(columns=["home__date", "home__team", "home__home", "home__season", "home__delta"])

# %%
df_g = pd.merge(
    df_g,
    df_aux.add_prefix("away__"),
    how="left",
    left_on=["date", "away"],
    right_on=["away__date", "away__team"]
).drop(columns=["away__date", "away__team", "away__home", "away__season", "away__delta"])

# %%
# just some cleaning
df_g.loc[df_g["home__val"] == "-", "home__val"] = 0
df_g.loc[df_g["away__val"] == "-", "away__val"] = 0
df_g.fillna(0, inplace=True)

# %%
df_g.to_csv(op.join(OUTPUT, "dataset.csv"), index=False)
# %%
