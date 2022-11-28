# --------------------------导包--------------------------

import re
from utils import *
from test import *
from transformers import AutoTokenizer
from Transformers.src.transformers.models.mt5.modeling_mt5 import  MT5ForConditionalGeneration
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

test_num = 2
set_seed(0)
train_dir = 'CCFnews/train_dataset.csv'
test_dir = 'CCFnews/test_dataset.csv'

train,test = load_data(train_dir,test_dir)
model_name = "mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)


summary, article_abstract = one_item(train,model,tokenizer,WHITESPACE_HANDLER)
print_rouge_L(summary, article_abstract)

train = many_item(train,model,tokenizer,WHITESPACE_HANDLER,test_num = test_num)
print_rouge_L(train["summary"][:test_num], train["Abstract"][:test_num])
# --------------------------预测-------------------------
test_eval(test,model,tokenizer,WHITESPACE_HANDLER,test_num=2)

