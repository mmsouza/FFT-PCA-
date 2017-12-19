import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition


def df_pca(normal_data, fault1_df, pcs, colNames):
    # Applying PCA
    full_df = normal_data.append(fault1_df, ignore_index=True)
    scaler = StandardScaler().fit(full_df)  # setup normalizer
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=colNames)  # formatting result as pandas dataframe
    full_df = full_df.sample(frac=1).reset_index(drop=True)  # data shuffle
    pca = decomposition.PCA(n_components=pcs)  # setting number of principle components
    pca.fit(full_df)

    # Applying pca and formatting as pandas dataframe
    names = []
    for j in range(0, pca.n_components):
        names.insert(len(names), "PC" + str(j))

    dfnormal = pd.DataFrame(data=pca.transform(normal_data), columns=names)
    dffailure = pd.DataFrame(data=pca.transform(fault1_df), columns=names)

    # Adding dummy data, labels that mark if a given occurrence is normal or a failure

    dffailure['failure'] = 1
    dfnormal['failure'] = 0

    # join both data classes
    full_df = dfnormal.append(dffailure, ignore_index=True)
    # full_df = full_df.sample(frac=1).reset_index(drop=True)
    # full_df=dffailure
    # Specify the data
    X = full_df.iloc[:, 0:pcs].astype(float)
    # Specify the target labels and flatten the array
    # y = np_utils.to_categorical(full_df.iloc[:, 13:14])
    y = full_df['failure']
    return X, y


