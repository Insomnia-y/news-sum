import csv
from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

IS_CNNDM = True # whether to use CNNDM dataset or XSum dataset
LOWER = False

# Load our model checkpoints
if IS_CNNDM:
    model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
    tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
else:
    model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
    tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

max_length = 1024 if IS_CNNDM else 512

def delete_number_before():
    with open("data/test_dataset.csv", mode="r", encoding="utf-8") as f:
        with open("data/test_dataset_no_number.csv", mode="w", encoding="utf-8") as f2:
            for row in f:
                pos = row.find('\t')
                newrow = row[pos+1::]
                print(newrow, file=f2, end='')
    print("in front of article numbers have been deleted!")

def solve(x):
    global model, tokenizer, max_length
    inputs = tokenizer([x], max_length=max_length, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"])
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def inference():
    with open("data/test_dataset_no_number.csv", mode="r", encoding="utf-8") as f:
        with open("data/output.csv", mode="w", encoding="utf-8") as f2:
            cnt = 0
            for row in f:
                row = solve(row)
                if cnt % 25 == 0:
                    print("{}/{} has been solved.".format(cnt, 1000))
                print("{}\t{}".format(cnt, row), file=f2)
                cnt += 1

if __name__ == '__main__':
    delete_number_before()
    inference()