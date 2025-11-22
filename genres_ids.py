import pandas as pd
import csv
import ast

genres = set()
ids = dict()
j = 0
IDS = "ids.csv"

with open("output.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        genre_list = ast.literal_eval(row[1])  # convert string to list
        for g in genre_list:
            genres.add(g)
            if g not in ids:
                ids[g] = j
                j += 1

# Convert dictionary to DataFrame
df = pd.DataFrame(list(ids.items()), columns=["genre", "id"])
df.to_csv(IDS, index=False)
