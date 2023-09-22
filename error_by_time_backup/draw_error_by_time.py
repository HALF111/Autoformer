import numpy as np
import matplotlib.pyplot as plt


def read_loss(file_name):
    loss_list = []
    with open(file_name) as f:
        losses = f.readlines()
        for loss in losses:
            # print(loss)
            loss_list.append(float(loss))
    return loss_list

# draw_fig
def draw_fig(dataset_name):
    loss_train = read_loss(dataset_name + "_train.txt")
    loss_vali = read_loss(dataset_name + "_vali.txt")
    loss_test = read_loss(dataset_name + "_test.txt")
    loss_before_adapt = read_loss(dataset_name + "_before_adapt.txt")
    loss_adapting = read_loss(dataset_name + "_adapting.txt")
    loss_after_adapt = read_loss(dataset_name + "_after_adapt.txt")

    # Draw Fig.1
    losses = loss_train + loss_vali + loss_test
    x = [i for i in range(len(losses))]
    print(len(losses), len(loss_train), len(loss_vali), len(loss_test))
    # print(losses)

    x_1 = [i for i in range(len(loss_train))]
    x_2 = [i for i in range(len(loss_train), len(loss_train)+len(loss_vali))]
    x_3 = [i for i in range(len(loss_train)+len(loss_vali), len(loss_train)+len(loss_vali)+len(loss_test))]

    # import matplotlib.pyplot as plt
    plt.figure()
    plt.title("loss function")
    plt.xlabel("running times")
    plt.ylabel("loss value")
    plt.plot(x_1, loss_train, label="loss_train", color='b')
    plt.plot(x_2, loss_vali, label="loss_vali", color='r')
    plt.plot(x_3, loss_test, label="loss_test", color='g')
    plt.plot(x_3, loss_after_adapt, label="loss_after_adapt", color='y')
    plt.legend(loc='best')
    plt.savefig(f"./figures/{dataset_name}_train_vali_test.pdf")
    # plt.show()


    # Draw Fig.2
    x = [i for i in range(len(loss_before_adapt))]
    print(len(x), len(loss_before_adapt), len(loss_adapting), len(loss_after_adapt))

    # import matplotlib.pyplot as plt
    plt.figure()
    # plt.title("loss function")
    # plt.xlabel("running times")
    # plt.ylabel("loss value")
    # plt.plot(x, loss_before_adapt, label="loss_before_adapt", color='b')
    # plt.plot(x, loss_adapting, label="loss_adapting", color='r')
    # plt.plot(x, loss_after_adapt, label="loss_after_adapt", color='g')
    diff = []
    for i in range(len(loss_before_adapt)):
        d = loss_before_adapt[i] - loss_after_adapt[i]
        diff.append(d)
    plt.hist(diff)
    # plt.legend(loc='best')
    plt.savefig(f"./figures/{dataset_name}_diff.pdf")
    # plt.show()

dataset_names = ["ETTh1", "ETTm2", "ECL", "Exchange", "Traffic", "Weather", "ILI"]
for dataset_name in dataset_names:
    draw_fig(dataset_name)