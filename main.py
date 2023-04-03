import os
import higher
import metrics
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
from pathlib import Path
from get_data import get_data
from methods.MetaViewer import MetaViewer
import warnings
warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('MetaViewer', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_set', default='CALTECH101_20', type=str)
    parser.add_argument('--dataset_location', default='datasets/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_false', default=True)
    # Model params
    parser.add_argument('--model', default='MetaViewer', type=str)
    parser.add_argument('--channels', nargs='+', type=int, help='auto encoder channels, like --channels -1 3 3 5')
    # Meta params
    parser.add_argument('--meta_channels', nargs='+', type=int, help='meta net channels, like --channels -1 3 3 5')
    parser.add_argument('--meta_kernels', default=3, type=int)
    parser.add_argument('--inner_step', default=2, type=int)
    parser.add_argument('--rate_support', default=0.4, type=float)
    # Training params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--validate_every', default=100, type=int)
    parser.add_argument('--inner_lr', default=1.0e-3, type=float)
    parser.add_argument('--outer_lr', default=1.0e-4, type=float)
    parser.add_argument('--decay_step', default=100, type=float)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--testing_path', default='')
    # save
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--output_str', default='')
    return parser


def bulid_metadata(data, rate_support=0.4):
    view_num = len(data)
    data_support = []
    data_query = []
    for v in range(view_num):
        num_data = data[v].shape[0]
        data_support.append(data[v][:int(num_data * rate_support), :])
        data_query.append(data[v][int(num_data * rate_support):, :])
    return data_support, data_query

def main(args):
    utils.print_args(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    print("Loading dataset ....")
    dataset_train = get_data(args.dataset_location, args.data_set, split='train')
    dataset_val = get_data(args.dataset_location, args.data_set, split='val')
    dataset_test = get_data(args.dataset_location, args.data_set, split='test')
    args.nb_classes = dataset_train.get_num_class()

    data_loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem, drop_last=True)
    sample_data, _ = next(iter(data_loader_train))
    print(args.nb_classes, len(sample_data))

    print(f"Creating model: {args.model} ...")
    model = MetaViewer(args=args, sample_data=sample_data)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        resume_path = args.output_dir + '/last_model.pth' if args.resume_path == '' else args.resume_path
        print('Resuming From: ', resume_path)
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state'], strict=True)
        args.epoch_start = checkpoint['epoch']

    train_x, train_y = dataset_train.get_full_data()
    train_x = [torch.Tensor(train_x[v]).to(device) for v in range(len(train_x))]
    if np.min(train_y) == 0:
        train_y += 1
    train_y = np.squeeze(train_y)

    val_x, val_y = dataset_val.get_full_data()
    val_x = [torch.Tensor(val_x[v]).to(device) for v in range(len(val_x))]
    if np.min(val_y) == 0:
        val_y += 1
    val_y = np.squeeze(val_y)

    test_x, test_y = dataset_test.get_full_data()
    test_x = [torch.Tensor(test_x[v]).to(device) for v in range(len(test_x))]
    if np.min(test_y) == 0:
        test_y += 1
    test_y = np.squeeze(test_y)

    acc_max = 0.
    if args.testing:
        print('Testing............')
        testing_path = args.output_dir + '/best_model.pth' if args.testing_path == '' else args.testing_path
        print('Testing From: ', testing_path)
        checkpoint = torch.load(testing_path, map_location='cpu')
        test_y = checkpoint['test_y']
        test_meta = checkpoint['test_meta']
        nb_classes = checkpoint['nb_classes']
        test_dict = {'Method': args.model, 'Data': args.data_set, 'Epoch': checkpoint['epoch']}
        test_dict.update(metrics.cluster(test_meta, test_y, nb_classes))
        utils.dict2json(os.path.join(args.output_dir, 'log_test.json'), test_dict, False)
        print('Test_epoch ', checkpoint['epoch'], 'clu_acc:', test_dict['clu_acc_avg'], 'clu_nmi:',
              test_dict['clu_nmi_avg'], 'clu_ar:', test_dict['clu_ar_avg'])

    else:
        print('Training...........')
        utils.dict2json(os.path.join(args.output_dir, 'log_train.json'), {'args': str(args)}, False)
        model.to(device)
        optimizer_inner = torch.optim.SGD(model.meta_net.parameters(), lr=args.inner_lr)
        optimizer_outer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
        outer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_outer, step_size=args.decay_step, gamma=0.9)

        for epoch in range(args.epoch_start, args.epochs):
            model.train()
            log_dict = {'epoch': epoch, 'outer_lr': optimizer_outer.state_dict()['param_groups'][0]['lr']}
            for idx, (data, targets) in enumerate(data_loader_train):
                data = [data[v].to(device).type(torch.cuda.FloatTensor) for v in range(len(data))]
                data_support, data_query = bulid_metadata(data, args.rate_support)

                optimizer_outer.zero_grad()
                with higher.innerloop_ctx(model, optimizer_inner, copy_initial_weights=False) as (
                        fmodel, diffopt):
                    # Inner level on support set
                    for step in range(args.inner_step):
                        for v in range(len(data_support)):
                            rec_support = fmodel.forward_base(data_support[v], v)
                            loss_inner = fmodel.loss_base(rec_support)
                            diffopt.step(loss_inner)
                    # Outer level on query set
                    for v in range(len(data_query)):
                        fmodel.s_enc[v].zero_grad()
                        fmodel.s_dec[v].zero_grad()
                    logits = fmodel.forward_meta(data_query, [v for v in range(len(data_query))])
                    loss_outer = fmodel.loss_meta(logits)
                    loss_outer.backward()
                optimizer_outer.step()

                log_dict['Loss_iter'+str(idx)] = loss_outer.item()
            outer_scheduler.step()
            utils.dict2json(os.path.join(args.output_dir, 'log_train.json'), log_dict, False)
            log_dict['state'] = model.state_dict()
            torch.save(log_dict, os.path.join(args.output_dir, "last_model.pth"))

            if epoch % args.validate_every == 0:
                model.eval()
                [_, _, train_meta, _] = model.forward_meta(train_x, [v for v in range(len(train_x))])
                [_, _, val_meta, _] = model.forward_meta(val_x, [v for v in range(len(val_x))])
                [_, _, test_meta, _] = model.forward_meta(test_x, [v for v in range(len(test_x))])

                test_dict = {'epoch': epoch}
                test_dict.update(metrics.cluster(val_meta, val_y, args.nb_classes))
                utils.dict2json(os.path.join(args.output_dir, 'log_val.json'), test_dict, False)
                print('Epoch ', epoch, 'clu_acc:', test_dict['clu_acc_avg'], 'clu_nmi:', test_dict['clu_nmi_avg'],
                      'clu_ar:', test_dict['clu_ar_avg'])

                if float(test_dict['clu_acc_avg']) >= acc_max:
                    log_dict['train_y'] = train_y
                    log_dict['test_y'] = test_y
                    log_dict['train_meta'] = train_meta
                    log_dict['val_meta'] = val_meta
                    log_dict['test_meta'] = test_meta
                    log_dict['nb_classes'] = args.nb_classes
                    torch.save(log_dict, os.path.join(args.output_dir, "best_model.pth"))
                    acc_max = float(test_dict['clu_acc_avg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MetaViewer', parents=[get_args_parser()])
    args = parser.parse_args()
    # build save path
    args.output_dir = 'checkpoints/' if args.output_dir == '' else args.output_dir
    args.output_dir = args.output_dir + args.model + '-' + args.data_set + '-C' + str(args.channels) + '-MC' + str(
        args.meta_channels) + '-lr[' + str(args.inner_lr) + ',' + str(args.outer_lr) + ']-' + str(
        int(args.decay_step)) + 'dy-' + str(args.batch_size) + 'bs-' + str(args.meta_kernels) + 'k-' + str(
        args.inner_step) + 'i-' + str(int(args.rate_support * 100)) + '%s' + args.output_str

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
