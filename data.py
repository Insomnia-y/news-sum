
train_path = './CCFNewsSummary/train_dataset.csv'  # 自定义训练集路径

with open(train_path, 'r', encoding='utf-8') as f:

    train_data_all = f.readlines()


test_path = './CCFNewsSummary/test_dataset.csv'  # 自定义测试集路径

with open(test_path, 'r', encoding='utf-8') as f:
    test_data = f.readlines()

    train = pd.DataFrame([], columns=["Index", "Text", "Abstract"])

    test = pd.DataFrame([], columns=["Index", "Text"])

    for idx, rows in enumerate(train_data_all):

        train.loc[idx] = rows.split("\t")

    for idx, rows in enumerate(test_data):

        test.loc[idx] = rows.split("\t")

