
a1, a2, a3 = [], [], []
# file_name = "ETTh1_96_24_Autoformer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
file_name = "ili_36_24_Autoformer_custom_ftM_sl36_ll18_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
with open(f"./loss_before_and_after_adapt/{file_name}.txt") as f:
    while True:
        # Get next line from file
        line = f.readline()
        # If line is empty then end of file reached
        if not line:
            break
        t1, t2, t3 = line.split(",")
        # print(t1, t2, t3)
        a1.append(float(t1))
        a2.append(float(t2))
        a3.append(float(t3))

print(sum(a1) / len(a1))
print(sum(a2) / len(a2))
print(sum(a3) / len(a3))