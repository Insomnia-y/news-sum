from tqdm import tqdm

def one_item(train,model,tokenizer,HANDLER):
    i = 0
    article_text = train["Text"][i]
    article_abstract = train["Abstract"][i]

    input_ids = tokenizer(
        [HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=512,
        min_length=int(len(article_text) / 32),
        no_repeat_ngram_size=3,
        num_beams=5
    )[0]
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(f"Generate：\n{summary}")
    print(f"Label：\n{article_abstract}")
    return summary, article_abstract


def many_item(train,model,tokenizer,HANDLER,test_num=5):
    for idx, article_text in tqdm(enumerate(train["Text"][:test_num]), total=test_num):
        input_ids = tokenizer(
            [HANDLER(article_text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=512,
            min_length=int(len(article_text) / 32),
            no_repeat_ngram_size=3,
            num_beams=5
        )[0]
        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        train.loc[idx, "summary"] = summary
    return train

def test_eval(test,model,tokenizer,HANDLER,device,test_num=1000):
    for idx, article_text in tqdm(enumerate(test["Text"]), total=test_num):
        input_ids = tokenizer(
            [HANDLER(article_text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=768
        )["input_ids"]
        output_ids = model.generate(
            input_ids=input_ids.to(device),
            max_length=512,
            min_length=int(len(article_text) / 32),
            no_repeat_ngram_size=3,
            num_beams=5
        )[0]
        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        test.loc[idx, "Text"] = summary
    test[["Index","Text"]].to_csv("T5summit01.csv",index=False,header=False,sep="\t")

