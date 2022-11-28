from src.transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-base")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
t = tok.batch_decode(generated_ids, skip_special_tokens=True)
print(t)