import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error as MSE

grads = np.load("./grads_test_small_ttn10.npy")
# grads = np.load("./grads_test_small_ttn10_sample0.10.npy")
print(grads.shape)

print(grads[0:5, 1:, 0])

grad_X = grads[:, 1:, :]
grad_Y = grads[:, 0:1, :]
print(grad_X.shape, grad_Y.shape)

# print(grad_X[0], grad_Y[0])

grad_X = grad_X.transpose(0, 2, 1)
grad_Y = grad_Y.transpose(0, 2, 1)
print(grad_X.shape, grad_Y.shape)
grad_X = grad_X.reshape(-1, grad_X.shape[-1])
grad_Y = grad_Y.reshape(-1, grad_Y.shape[-1])
print(grad_X.shape, grad_Y.shape)

grad_X_mean = grad_X.mean(axis=1)
print(grad_X_mean.shape)


LR_model = LinearRegression(fit_intercept=False)
LR_model.fit(grad_X, grad_Y)

yhat = LR_model.predict(grad_X)

print(LR_model.score(grad_X, grad_Y))
print(LR_model.coef_)
print(LR_model.intercept_)

mse_predict = MSE(yhat, grad_Y)
mse_mean = MSE(grad_X_mean, grad_Y)
print(f"mse_predict is: {mse_predict}")
print(f"mse_mean is: {mse_mean}")
print(grad_Y.mean(), yhat.mean())

print(type(LR_model.coef_))
print(LR_model.coef_.shape)
print(LR_model.coef_[0].shape)

print(type(LR_model.coef_[0].tolist()))
print(LR_model.coef_[0].tolist())
print(len(LR_model.coef_[0].tolist()))


# class LR(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(10, 1, bias=False)
    
#     def forward(self, x):
#         out = self.linear(x.permute(1,0)).permute(1,0)
#         return out

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# grad_X, grad_Y = torch.tensor(grad_X).to(device), torch.tensor(grad_Y).to(device)

# # model = LR()
# # print(model.linear.weight)
# # print(model.linear.bias)
# # print(grad_X[0].shape)
# # print(model(grad_X[0]).shape)
# # print(grad_Y[0].shape)
# # print(model(grad_X[0]).shape == grad_Y[0].shape)


# model = LR().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-6)

# last_loss = float("inf")

# for epoch in range(10):
#     print(f"epoch: {epoch+1}")
#     total_loss = 0
#     for i in range(grad_X.shape[0]):
#         grad_X_sample = grad_X[i]
#         grad_Y_sample = grad_Y[i]
        
#         out = model(grad_X_sample)
#         loss = criterion(out, grad_Y_sample)
#         total_loss += loss.item()

#         loss.backward()
#         optimizer.step()

#         if (i+1) % 500 == 0:
#             print(f"epoch:{epoch+1}, iteration:{i+1}, product loss:{loss.item()}")
    
#     print(total_loss)
#     if total_loss > last_loss:
#         print("Early stopping")
#         break
#     last_loss = total_loss

# print(model.linear.weight)
# print(model.linear.bias)

