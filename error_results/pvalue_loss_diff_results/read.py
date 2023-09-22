import matplotlib.pyplot as plt

file_name = "ili.txt"

p_value_list, loss_diff_list = [], []
with open(file_name) as f:
    lines = f.readlines()
    for line in lines:
        p_value, loss_diff = line.split(",")
        p_value, loss_diff = float(p_value), float(loss_diff)
        p_value_list.append(p_value)
        loss_diff_list.append(loss_diff)
        print(loss_diff)

print(len(p_value_list))
print(len(loss_diff_list))
print(f"Average of loss_diff: {sum(loss_diff_list)/len(loss_diff_list)}")

