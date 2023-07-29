import numpy as np

lookback_data_npz = "./data_test_ttn10.npz"

# 再从npz中获取回来
npzfile = np.load(lookback_data_npz)
all_data_x = npzfile["arr_0"]
all_data_y = npzfile["arr_1"]

test_x, test_y = all_data_x[:, 0:1, :], all_data_y[:, 0:1, :]
lookback_x, lookback_y = all_data_x[:, 1:, :], all_data_y[:, 1:, :]

print(test_x.shape)
print(test_y.shape)
print(lookback_x.shape)
print(lookback_y.shape)

lookback_y = lookback_y[:, :, -96:, :]
print(lookback_y.shape)


print(test_x[0, 0, :20, 1])
print(test_x[1, 0, :20, 1])
print(test_x[2, 0, :20, 1])