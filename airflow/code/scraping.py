#%%
from operator import methodcaller
import os.path as op
import re
import argparse
from functools import partial
import requests
from bs4 import BeautifulSoup
import pandas as pd
from toolz import compose, excepts
import numpy as np

# %%
HEADERS = {'User-Agent':
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
# %%
def team_value_for_year(year):
    """
    Obtains all the teams' value for a given year
    """
    assert int(year) >= 2011, "Years can only be from 2011 onwards"
    year = str(year)
    date = f"{year}-08-01"
    page = f"https://www.transfermarkt.com/liga-nos/marktwerteverein/wettbewerb/PO1/plus/?stichtag={date}"
    pageTree = requests.get(page, headers=HEADERS)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    club_names = [
        x.text.strip() for x in
        pageSoup.select("table.items tr td:nth-child(3)") if x.text.strip()
    ][1:]
    club_value = [
        x.text.strip() for x in
        pageSoup.select("table.items tr td:nth-child(5)") if x.text.strip()
    ][1:]

    df = pd.DataFrame(club_names, columns=["name"])
    df["val"] = club_value
    df["val"] = df["val"].apply(
        lambda x: re.sub(r"(.+)\,(\d\d) mil. €", r"\g<1>\g<2>0", x)
    ).apply(
        lambda x: re.sub(" K €", "00", x)
    )
    df["year"] = year
    return df

#%%
def games_for_year(year, future):
    """
    Obtains all the games results for a given year
    """
    year = str(year)
    page = f"https://www.transfermarkt.com/liga-nos/gesamtspielplan/wettbewerb/PO1/saison_id/{year}"
    pageTree = requests.get(page, headers=HEADERS)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
    tables = pageSoup.select("div.box > table")
    cln_dates = compose(
        methodcaller("strip"),
        partial(re.sub, "\t+", " "),
        partial(re.sub, r"(\r\n.+)?\d+:\d\d \w\w", " "),
        lambda x: x.text.strip()
    )

    cln_teams = compose(
        partial(re.sub, fr"(\(.+\)|\xa0)", ""),
        lambda x: x.text.strip()
    )
    dfs = []
    for t in tables:
        dates = *map(cln_dates, t.select("td:nth-child(1):not(.show-for-small)")),
        df = pd.DataFrame(dates, columns=["date"])
        df[df.date.apply(len) == 0] = np.nan
        df.date.fillna(inplace=True, method="ffill")
        home_teams = *map(cln_teams, t.select("td:nth-child(3)")),
        df["home"] = home_teams
        away_teams = *map(cln_teams, t.select("td:nth-child(7)")),
        df["away"] = away_teams
        result = *map(lambda x: x.text.strip(), t.select("td:nth-child(5)")),
        df["result"] = result
        df["goals_home"] = df.result.apply(lambda r: r.split(":")[0])
        df["goals_away"] = df.result.apply(lambda r: r.split(":")[1])
        df.drop(columns="result", inplace=True)
        df["season"] = year
        if future and len(df.query("goals_home == '-'")) > 0:
            return df.query("goals_home == '-'").reset_index(drop=True)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    return df.query(
        "goals_home != '-'"
    ).reset_index(drop=True)
# %%
INTERACTIVE = excepts(
    Exception,
    lambda: get_ipython() == get_ipython(), lambda _: False
)()
YEAR = "2019"
OUTPUT = "env_raw/data"
FUTURE = False
# %%
if not INTERACTIVE:
    parser = argparse.ArgumentParser(
        description='Scrapes transfermarkt.com for games and team value.'
    )
    parser.add_argument(
        '-o',
        "--output",
        help='The output where the scraping result is to be saved',
        default=OUTPUT
    )
    parser.add_argument(
        "year"
    )
    parser.add_argument(
        "-f",
        "--future",
        help="Scrape only future games",
        action="store_true",
        default=FUTURE
    )

    args = parser.parse_args()
    YEAR = args.year
    OUTPUT = args.output
    FUTURE = args.future
    print(f"Will scrape for year {YEAR}")

if not FUTURE:
    path = op.join(OUTPUT, "values.csv")

    df = team_value_for_year(YEAR)
    try:
        df_old = pd.read_csv(path)
        df = pd.concat([df, df_old])
        df = df.drop_duplicates()
    except:
        print("Could not open previous values file")
    #df.to_csv(path, index=False)

df = games_for_year(YEAR, future=FUTURE)

path = op.join(OUTPUT, "upcoming_games.csv" if FUTURE else "games.csv")
if not FUTURE:
    try:
        df_old = pd.read_csv(path)

        df = pd.concat([df, df_old])

        df = df.drop_duplicates()
    except:
        print("Unable to read previous file")
#%%
print(f"Will output {len(df)} to {path}")
#%%
df.to_csv(path, index=False)





# %%
