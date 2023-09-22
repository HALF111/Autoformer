import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = "national_illness"
file_before = f"../error_by_time/{name}_before_adapt.txt"
file_after = f"../error_by_time/{name}_after_adapt.txt"

name = "ili"
file_pvalue = f"./{name}_ks_test.txt"

loss_before, loss_after = [], []

# 1. loss_before
with open(file_before) as f:
    lines = f.readlines()
    for line in lines:
        loss = float(line)
        loss_before.append(loss)

# 2. loss_after
with open(file_after) as f:
    lines = f.readlines()
    for line in lines:
        loss = float(line)
        loss_after.append(loss)
        
# calculate diff of loss
loss_diff = []
for i in range(len(loss_before)):
    loss_diff.append(loss_before[i] - loss_after[i])

# 3. p_value
p_value_list = []
with open(file_pvalue) as f:
    lines = f.readlines()
    for line in lines:
        statistic, p_value = line.split(",")
        statistic, p_value = float(statistic), float(p_value)
        p_value_list.append(p_value)

print(len(p_value_list), len(loss_diff))

result_dir = "pvalue_loss_diff_results"
file_name = result_dir + "/" + name + ".txt"
with open(file_name, "w") as f:
    for i in range(len(p_value_list)):
        f.write(f"{p_value_list[i]}, {loss_diff[i]}\n")
        print(loss_diff[i])

print(f"Average of loss_diff: {sum(loss_diff)/len(loss_diff)}")

plt.figure()
plt.scatter(p_value_list, loss_diff)
plt.savefig(f"{name}_pvalue_loss_diff.pdf")