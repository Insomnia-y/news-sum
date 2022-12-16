import time
from tqdm import tqdm
pbar=tqdm(range(10))
for i in pbar:
    time.sleep(1)     # your code here
    pbar.set_description("iter %d"%i)