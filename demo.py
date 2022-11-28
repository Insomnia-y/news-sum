import re
from train import *
from utils import *
from datas import *
import csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from Transformers.src.transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
devicenum=1
device = torch.device("cuda" + ":" + str(devicenum) if torch.cuda.is_available() else "cpu")
set_seed(12345)
save_model = True
generate_test = True
to_train = True
to_valid = True
learning_rate = 2e-5
epoch_num = 10
max_input_length=512
max_target_length = 64
train_dir = 'CCFNewsSummary/train.txt'
valid_dir = 'CCFNewsSummary/valid.txt'
test_dir = "CCFNewsSummary/test_dataset.csv"
# prepare model
model_name = "mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

checkpoint = torch.load("checkpoints/epoch_9_valid_rouge_74.1795_model_weights.bin", map_location=lambda storage, loc: storage.cuda(device))
model.load_state_dict(checkpoint, strict=False)

# prepare data
train_data = NewsDataset(train_dir)
valid_data = NewsDataset(valid_dir)
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=DataCollator(tokenizer,model,max_input_length,max_target_length,True))
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=DataCollator(tokenizer,model,max_input_length,max_target_length,False))

#prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)


total_loss = 0.
best_avg_rouge = 0.
for t in range(epoch_num):
    if not to_train and t>0:
        break
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    if to_train:
        total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss,device)
    if to_valid:
        valid_rouge = test_loop(valid_dataloader, model,tokenizer,device,max_target_length)
        print(valid_rouge)
        rouge_avg = valid_rouge['avg']
        if rouge_avg > best_avg_rouge:
            best_avg_rouge = rouge_avg
            if save_model:
                print('saving new weights...\n')
                torch.save(model.state_dict(), f'checkpoints/epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
print("Done!")

if generate_test:
    test_data = NewsevalDataset(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=DataevalCollator(tokenizer,model))
    test = data_eval(test_dataloader, model,tokenizer,device,max_target_length=64)
    with open("CCFNewsSummary/submission.csv","w",encoding='utf-8', newline='')as f:
        for i in range(1000):
            f.write(str(test["ID"][i])+"\t"+test["Title"][i]+"\n")
        f.close()
