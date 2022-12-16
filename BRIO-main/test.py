import time
from tqdm import tqdm

def delete_number_before():
    with open("data/test_dataset.csv", mode="r", encoding="utf-8") as f:
        with open("data/test_dataset_no_number.csv", mode="w", encoding="utf-8") as f2:
            # print("len(f) = {}".format(len(f)))
            pbar = tqdm(iterable=f, total=1000)
            for row in pbar:
                time.sleep(0.1)
                pos = row.find('\t')
                newrow = row[pos+1::]
                print(newrow, file=f2, end='')
    print("in front of article numbers have been deleted!")

def insert_endl():
    with open("data/output.csv", mode="r", encoding="utf-8") as f:
        with open("data/submit.csv", mode="w", encoding="utf-8") as f2:
            str = f.readline()
            prepos = 0
            for i in range(1, 1000):
                now = "{}\t".format(i)
                pos = str.find(now)
                print(str[prepos:pos], file=f2)
                prepos = pos
            print(str[pos:-1], file=f2)

def process_submit_punctuation():
    with open("data/submit.csv", mode="r", encoding="utf-8") as f:
        with open("data/submit2.csv", mode="w", encoding="utf-8") as f2:
            pbar = tqdm(iterable=f, total=1000)
            cnt = 0
            for row in pbar:
                pos = row.find('\t')
                row2 = row[pos + 1 :: ]
                row2 = row2.replace("\'s ", " \'s ")
                row2 = row2.replace("  \'s ", " \'s ")
                row2 = row2.replace(", ", " , ")
                row2 = row2.replace(": ", " : ")
                row2 = row2.replace("; ", " ; ")
                row2 = row2.replace("? ", " ? ")
                row2 = row2.replace("/", " / ")
                row2 = row2.replace(" \"", " \'\' ")
                row2 = row2.replace("\" ", " \'\' ")
                row2 = row2.replace(". ", " . ")
                row2 = row2.replace(".\n", " .\n")
                row2 = row2.replace("u.s . ", "u.s. ")
                row2 = row2.replace("u.s .\n", "u.s.\n")
                print("{}\t{}".format(cnt, row2), file=f2, end='')
                cnt += 1



if __name__ == '__main__':
    # delete_number_before()
    # insert_endl()
    process_submit_punctuation()