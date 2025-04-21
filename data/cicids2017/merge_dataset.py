import os
import pandas as pd

if __name__ == "__main__":

    files = [f for f in os.listdir("./data/cicids2017") if f in ('monday', 'tuesday', 'wednesday', 'thursday', 'friday')]

    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)

    df.sort_values("Timestamp", inplace=True)
    df.to_csv("./data/cicids2017/cicids2017.csv", index=False) 