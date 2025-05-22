import os
import runpy
from contextlib import contextmanager
import numpy as np
os.environ["MKL_THREADING_LAYER"] = "GNU"
import csv
import subprocess, sys
import time

models_to_run = [#"Qwen/Qwen2.5-1.5B-Instruct", 
    #"Qwen/Qwen2.5-3B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]

@contextmanager
def chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

# path: [eval_file, mean_acc]
data_dict = {"biography": ["gen_conversation.py", "eval_conversation.py"],
             "math": ["gen_math.py"],
             "gsm": ["gen_gsm.py", "eval_gsm.py"],
             "mmlu": ["gen_mmlu.py", "eval_mmlu.py"],
             }

# gen a bunch of results 
# for each, run eval
for model in models_to_run:
    print("Starting VLLM, Model:", model)
    proc = subprocess.Popen(
        ["vllm", "serve", model],
        text=True,
    )
    time.sleep(60)
    for path, eval_file in data_dict.items():
        with chdir(path):
            print("Running: " + path)
            if path != "math":
                runpy.run_path(eval_file[0], run_name="__main__", init_globals={"MODEL": model})
                print("Evaluating: " + path)
                mean_acc = np.mean(runpy.run_path(eval_file[1], run_name="__main__", init_globals={"MODEL": model}).get('accuracies'))
            # Math: does not have a separate eval file; score within gen file
            elif path == "math":
                print("Evaluating: " + path)
                mean_acc = np.mean(runpy.run_path(eval_file[0], run_name="__main__", init_globals={"MODEL": model}).get('scores'))
        print("Appending mean accuracy...")
        data_dict[path].append(mean_acc)

    print("Stopping VLLM")
    proc.terminate()
    proc.wait()
    print("VLLM stopped:", proc.returncode)

    acc_list = [model] + [float(i[-1]) for i in data_dict.values()]

    print("Writing to CSV")
    with open("accuracies.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)