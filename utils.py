import torch
from rouge import Rouge
import pandas as pd
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
# 评测 rouge_L 分数
def print_rouge_L(output, label):
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, label)
    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))

def load_data(train_dir,test_dir):
    with open(train_dir, 'r', encoding='utf-8') as f:
        train_data_all = f.readlines()
        f.close()
    with open(test_dir, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
        f.close()
    train = pd.DataFrame([], columns=["Index", "Text", "Abstract"])
    test = pd.DataFrame([], columns=["Index", "Text"])
    for idx, rows in enumerate(train_data_all):
        train.loc[idx] = rows.split("\t")
    for idx, rows in enumerate(test_data):
        test.loc[idx] = rows.split("\t")
    return train,test