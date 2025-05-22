import os
import runpy
from contextlib import contextmanager
import numpy as np
import csv

CURRENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

@contextmanager
def chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

# path: [eval_file, mean_acc]
data_dict = {"biography": ["eval_conversation.py"],
             "gsm": ["eval_gsm.py"],
             "mmlu": ["eval_mmlu.py"]}

# gen a bunch of results 
# for each, run eval

for path, eval_file in data_dict.items():
    with chdir(path):
        print("Evaluating: " + path)
        mean_acc = np.mean(runpy.run_path(eval_file[0], run_name="__main__").get('accuracies'))
    print("Appending mean accuracy...")
    data_dict[path].append(mean_acc)

acc_list = [CURRENT_MODEL] + [float(i[1]) for i in data_dict.values()]

print("Writing to CSV")
with open("accuracies.csv", "w", newline="") as f:
    writer = csv.writer(f)
    row_to_write = [acc_list]
    writer.writerows(row_to_write)