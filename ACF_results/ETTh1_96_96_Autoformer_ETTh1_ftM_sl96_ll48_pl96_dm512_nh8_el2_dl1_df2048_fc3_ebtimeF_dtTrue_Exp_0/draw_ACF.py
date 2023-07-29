import numpy as np
import matplotlib.pyplot as plt

lag = 1
name = "ETTh1"

acf_train = np.load(f"lag{lag:02d}_train.npy")
acf_val = np.load(f"lag{lag:02d}_val.npy")
acf_test = np.load(f"lag{lag:02d}_test.npy")
print(acf_train.shape, acf_val.shape, acf_test.shape)


x_1 = [i for i in range(len(acf_train))]
x_2 = [i for i in range(len(acf_train), len(acf_train) + len(acf_val))]
x_3 = [i for i in range(len(acf_train) + len(acf_val), len(acf_train) + len(acf_val) + len(acf_test))]

x = [i for i in range(len(acf_train) + len(acf_val) + len(acf_test))]

plt.title(f"{name} dataset overview")
plt.xlabel("timestamp")
plt.ylabel("acf_value")
# channel = 0
# plt.plot(x_1, acf_train[:, channel], label='train', color='b')
# plt.plot(x_2, acf_val[:, channel], label='vali', color='r')
# plt.plot(x_3, acf_test[:, channel], label='test', color='g')
for channel in range(acf_train.shape[1]):
    plt.plot(x, np.concatenate((acf_train[:, channel], acf_val[:, channel], acf_test[:, channel])), label=f"channel{channel:02d}")
plt.axvline(x=len(acf_train))
plt.axvline(x=len(acf_train)+len(acf_val))
plt.legend(loc='best')

plt.savefig(f"./lag{lag:02d}.pdf")
plt.show()