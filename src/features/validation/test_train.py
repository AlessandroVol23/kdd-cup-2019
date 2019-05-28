def test_train_split_df(df_r, test_percentage=0.25):
    import numpy as np
    df = df_r.copy()

    msk = np.random.rand(len(df)) < (1-test_percentage)

    train = df[msk]
    test = df[~msk]

    return train, test
