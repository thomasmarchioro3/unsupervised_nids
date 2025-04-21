import pandas as pd

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("./data/cicids2017/cicids2017.csv")
    df_sample, _ = train_test_split(df, train_size=0.1, random_state=42, stratify=df["Label"])
    df_sample.sort_values("Timestamp", inplace=True)
    df_sample.to_csv("./data/cicids2017/cicids2017_random10.csv", index=False)