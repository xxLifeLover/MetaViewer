import json


def dict2json(file_name, res, flag=True):
    if flag:
        print(res)
    with open(file_name, 'a+') as json_file:
        json.dump(res, json_file)
        json_file.write('\n')


def print_args(args):
    print('***********************************************')
    print('Model: ', args.model)
    print('Dataset: ', args.data_set)
    print('Dataset Location: ', args.dataset_location)
    if args.resume:
        print('From ', args.resume_path, 'to ', args.output_dir)
    else:
        print('To ', args.output_dir)
    print('---------- Network ----------')
    print('Channels: ', args.channels)
    if 'Meta' in args.model:
        print('Meta Channels: ', args.meta_channels)
        print('Meta Kernels: ', args.meta_kernels)
        print('Inner Lr: ', args.inner_lr)
        print('Outer Lr: ', args.outer_lr)
        print('S/Q: ', args.rate_support * 10, ':', (1 - args.rate_support) * 10)
    print('---------- Training ----------')
    print('Batchsize: ', args.batch_size)
    print('Epochs: ', args.epochs)


def deal(list_ori, p):
    list_new = []
    list_short = []
    for k, v in enumerate(list_ori):
        if v == p and k != 0:
            list_new.append(list_short)
            list_short = []
        list_short.append(v)
    list_new.append(list_short)
    return list_new