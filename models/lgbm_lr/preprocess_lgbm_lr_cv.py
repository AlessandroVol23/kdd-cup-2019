import pandas as pd

def load_data(path):
    '''
    loads the kdd cup data and a column "target" for ranking

    :param path: str
                path to the pickle file
    :return: pandas data frame
                with the data
    '''
    print("load_data(" + str(path) + ") ...")

    # path to raw pickle_file
    data = pd.read_pickle(path)
    # print(list(data.columns.values))
    # print(data.head(10))

    # sort data to sid for mapping to the query file
    data.sort_values(by="sid", inplace=True)
    # print(data.head(10))

    # drop same transport_mode in every session (sid)
    data.drop_duplicates(["sid", "transport_mode"], inplace=True)
    # print(data.head(10))

    # reindexing
    data.reset_index(drop=True, inplace=True)
    # print(data.head(10))

    # assign new column target to the data
    if "click_mode" in data:
        data = data.assign(target=data.apply(lambda x: 3 if x.click_mode == x.transport_mode else 0, axis=1))
        # print(data.head(10))

    print("data loaded !!!")
    return data

def load_sid_folds(path):
    '''
    load the SID folds for cv

    hardcoded file names: "SIDs_<i>.txt"

    :param path: str

    :return: list with pandas data frames with the SIDs for the 5cv folds
    '''
    print("load_sid_folds(" + str(path) + ") ...")

    # init empty list
    sid_folds = []

    # walk through to the five sid pickles and add it to the list
    for i in range(1, 6):
        file = 'SIDs_' + str(i) + '.txt'
        print("round : " + str(i) + " with " + file)
        sid_folds.append(pd.DataFrame(pd.read_pickle(file), columns=['sid'], ).sort_values('sid').reset_index(drop=True))
    print("sid_folds loaded !!!")

    # return list
    return sid_folds


def create_data_folds(data,sid_folds):
    '''
    does a join with the data and SID folds

    :param data: pandas data frame
                for the kdd cup

    :param sid_folds: list
                with sid folds pandas data frames

    :return: list
                with pandas data frames for cv
    '''
    print("create data_folds(...) ...")

    # init empty list
    data_folds = []

    # walk through the folds and join them with the data by the SIDs
    for i in sid_folds:
        #print(i.shape)
        data_folds.append(i.join(data.set_index('sid'), 'sid'))

    #for i in data_folds:
        #print(i.shape)

    print("data_folds created !!!")

    # return list
    return data_folds

def save_data_folds(data_folds):
    '''
    save pickle files for cv

        lgbm_lr_f<i>.pickle for the i-th fold
        lgbm_lr_fout<i>.pickle for the other folds for i

    :param data_folds: list
                with pandas data frames for cv
    :return:
    '''
    print("save_data_folds(...) ...")

    # build data
    for i in range(0, 5):

        file_name = str("lgbm_lr_f" + str(i) + ".pickle")
        data_folds[i].to_pickle(file_name)
        print(str(file_name) + " saved!")

        t_data = []
        for j in range(0, 5):
            if i != j:
                print("round: " + str(i) + " " + str(j))
                t_data.append(data_folds[j])

        file_name = str("lgbm_lr_fout" + str(i) + ".pickle")
        pd.concat(t_data).to_pickle(file_name)
        print(str(file_name) + " saved!")

    print("data_folds saved !!!")


# 1 load data
# 2 load SID folds
# 3 create data folds
# 4 save it in actual directory

save_data_folds(create_data_folds(load_data("../data/processed/processed_all/processed_all/train_all_row.pickle"),load_sid_folds("")))

