import pandas as pd
import common
import prince

from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# scaler = MinMaxScaler()
data = pd.read_csv("VehiclesDBV2.csv")

categorical_cols = pd.read_csv("selected_categorical_columns_df.csv")
categorical_cols = categorical_cols[categorical_cols['described variance'] > 0.25]['selected'].to_list()

data[common.numerical_cols] = data[common.numerical_cols].fillna(-1)
data[common.categorical_cols] = data[common.categorical_cols].fillna("_")

data['BRAND_INTERNET_ADDRESS_DS'] = data['BRAND_INTERNET_ADDRESS_DS'].str.split(".").str[-1]
data['VALIDITY_DT'] = data['VALIDITY_DT'].astype(str).str[:4]

# data[common.numerical_cols] = scaler.fit_transform(data[common.numerical_cols])

# data_train, data_test = train_test_split(data, test_size=0.15, random_state=42)
# print(f"{data_train.shape[0]} training observations and {data_test.shape[0]} observations")

selected_categorical_columns = []
selected_categorical_columns_variance = []

pca = prince.PCA(n_components=21,
                 n_iter=10,
                 copy=True,
                 check_input=True,
                 engine='fbpca')

famd_fit = pca.fit(data[common.numerical_cols])
print(f"explained variance by PCA for numerical data = {famd_fit.explained_inertia_.sum():.3f}")

mca = prince.MCA(n_components=len(categorical_cols),
                 n_iter=10,
                 copy=True,
                 check_input=True,
                 engine='fbpca')

mca_fit = mca.fit(data[categorical_cols])
print(f"explained variance by MCS for categorical data = {sum(mca_fit.explained_inertia_):.3f}")

datanumerical_transformed = famd_fit.row_coordinates(data[common.numerical_cols])
data_categorical_transformed = mca_fit.row_coordinates(data[categorical_cols])

# datanumerical_transformed.to_csv("data_numerical_transformed.csv")
# data_categorical_transformed.to_csv("data_categorical_transformed.csv")
# data.to_csv("data.csv")