import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name_pure = "angels_allones_ttn10_lambda500.00"
file_name = f"{file_name_pure}.txt"

angels, loss_before, loss_after = [], [], []

with open(file_name) as f:
    lines = f.readlines()
    for line in lines:
        angel, loss1, loss2 = line.split(",")
        angel, loss1, loss2 = float(angel), float(loss1), float(loss2)
        angels.append(angel); loss_before.append(loss1); loss_after.append(loss2)

print(len(angels))

loss_diff = []
for i in range(len(loss_before)):
    loss_diff.append(loss_before[i] - loss_after[i])

plt.figure()
plt.scatter(angels, loss_diff)
plt.savefig(f"{file_name_pure}.pdf")