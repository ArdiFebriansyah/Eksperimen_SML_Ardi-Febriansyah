
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(input_path):
    df = pd.read_csv(input_path)
    return df


def clean_data(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.drop_duplicates()

    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def preprocess_data(df):
    df = clean_data(df)

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop(columns=["num"])
    y = df_encoded["num"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    df_preprocessed = pd.concat(
        [X_scaled_df, y.reset_index(drop=True)],
        axis=1
    )

    return df_preprocessed


def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    input_path = "../heart_raw/heart.csv"
    output_path = "heart_preprocessing/heart_preprocessed.csv"

    df = load_data(input_path)
    df_preprocessed = preprocess_data(df)
    save_data(df_preprocessed, output_path)

    print("Preprocessing berhasil.")
    print(f"Dataset hasil preprocessing disimpan di: {output_path}")
    print(f"Jumlah data: {df_preprocessed.shape[0]}")
    print(f"Jumlah kolom: {df_preprocessed.shape[1]}")


if __name__ == "__main__":
    main()
