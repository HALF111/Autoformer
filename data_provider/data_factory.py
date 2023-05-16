from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data_provider.data_loader import Dataset_ETT_hour_Test, Dataset_Custom_Test, Dataset_ETT_minute_Test
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False  # 测试集无需做随机shuffle
        # drop_last = False
        drop_last = True  # 设置成True还是False？
        batch_size = args.batch_size
        freq = args.freq
    elif flag == "test_with_batchsize_1":
        flag = "test"
        shuffle_flag = False
        drop_last = True  # 设置成True还是False？
        batch_size = 1
        freq=args.freq
    elif flag == "val_with_batchsize_1":
        flag = "val"
        shuffle_flag = True
        drop_last = True  # 设置成True还是False？
        batch_size = 1
        freq=args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:  # flag == "train" or flga == "val"
        shuffle_flag = True  # 注意：训练集/验证集数据是会随机shuffle的！！
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def data_provider_at_test_time(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        # drop_last = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

        # 注意：因为我们要做TTT/TTA，所以一定要把batch_size设置成1 ！！！
        # batch_size = 32
        batch_size = args.adapted_batch_size
        # batch_size = 1
        # batch_size = 4
        # batch_size = 2857
        # batch_size = 256

        # Data = Dataset_Custom_Test
        dataset_name = args.data
        if dataset_name == "ETTh1" or dataset_name == "ETTh2":
            Data = Dataset_ETT_hour_Test
        elif dataset_name == 'ETTm1' or dataset_name == 'ETTm2':
            Data = Dataset_ETT_minute_Test
        else:
            Data = Dataset_Custom_Test


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        test_train_num = args.test_train_num
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader
