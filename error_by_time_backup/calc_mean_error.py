import numpy as np
import os

a = []

paths = os.walk("./")
for path, dir_lst, file_lst in paths:
    for file_name in file_lst:
        # print(file_name)
        if 'txt' in file_name:
            source_file = os.path.join(path, file_name)
            print(source_file)

            with open(source_file) as f:
                while True:
                    # Get next line from file
                    line = f.readline()
                    # If line is empty then end of file reached
                    if not line:
                        break
                    # t1, t2, t3 = line.split(",")
                    t = float(line)
                    # print(t1, t2, t3)
                    a.append(float(t))

            print(sum(a) / len(a))




# loss_lst, grad_lst = [], []
# # file_name = "ETTh1_96_24_Autoformer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
# file_name = "batch128_Exchange_96_96_Autoformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
# with open(f"./logs_loss_and_grad/{file_name}") as f:
#     while True:
#         # Get next line from file
#         line = f.readline()
#         # If line is empty then end of file reached
#         if not line:
#             break
#         t1, t2 = line.split(",")
#         # print(t1, t2)
#         loss_lst.append(float(t1))
#         grad_lst.append(float(t2))

# print(len(loss_lst))
# print(np.mean(loss_lst))
# print(np.var(loss_lst))
# print(np.std(loss_lst, ddof=1))
# print(np.mean(grad_lst))
# print(np.var(grad_lst))
# print(np.std(grad_lst, ddof=1))
