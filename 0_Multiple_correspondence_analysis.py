import pandas as pd
import common
import prince

from sklearn.preprocessing import MinMaxScaler

import tqdm

scaler = MinMaxScaler()
data = pd.read_csv("VehiclesDBV2.csv")

data[common.categorical_cols] = data[common.categorical_cols].fillna("_")

data['BRAND_INTERNET_ADDRESS_DS'] = data['BRAND_INTERNET_ADDRESS_DS'].str.split(".").str[-1]
data['VALIDITY_DT'] = data['VALIDITY_DT'].astype(str).str[:4]

selected_categorical_columns = []
selected_categorical_columns_variance = []

for iter in range(len(common.categorical_cols)):
    max_explained_variance = 0
    for col in tqdm.tqdm(common.categorical_cols):
        if col not in selected_categorical_columns:
            famd = prince.MCA(n_components=1+len(selected_categorical_columns),
                              n_iter=10,
                              copy=True,
                              check_input=True,
                              engine='fbpca')

            famd_fit = famd.fit(data[[col] + selected_categorical_columns])

            explained_var = sum(famd_fit.explained_inertia_)

            if explained_var > max_explained_variance:
                max_explained_variance = max(max_explained_variance, explained_var)
                selected_categorical_column = col
    selected_categorical_columns.append(selected_categorical_column)
    selected_categorical_columns_variance.append(max_explained_variance)
    print(
        f"explained variance = {max_explained_variance:.2f} with {len(selected_categorical_columns)} selected categorical vars")

selected_categorical_columns_df = pd.DataFrame(
    list(map(list, zip(*[selected_categorical_columns, selected_categorical_columns_variance]))),
    columns=["selected", "described variance"])
selected_categorical_columns_df.to_csv("selected_categorical_columns_df.csv")
