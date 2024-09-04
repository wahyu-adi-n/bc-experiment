from activation import activation_dict
from config import GAMMA, SEED, STEP_SIZE, INV_LABEL_DICTS, MEAN, STD
from datasets import num_classes_dict
from datetime import datetime
from networks import network_dict
from visualization import plot_confusion_matrix, plot_acc_loss_curve
from torch.utils.data import DataLoader
from torch import nn, optim
# from torchinfo import summary
from torchvision import transforms as T
from train_eval import do_train, do_eval

import argparse
import datasets
import os
import pandas as pd
import time
import torch

gamma = GAMMA
seed = SEED
step_size = STEP_SIZE

def main(args):
    assert args.best_metric in ['acc', 'auroc', 'f1', 'precision', 'recall'], 'best metric must be one of acc, auroc, f1, precision, recall'
    
    # random.seed(seed)     # python random generator
    # np.random.seed(seed)  # numpy random generator

    # reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform = T.Compose(
            [
                T.Resize((256, 256), antialias=True),  # default => (460, 700)
                T.CenterCrop(size=(224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),
                # T.Normalize((0.5,), (0.5,)),
                # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet normalization
                T.Normalize(mean=MEAN, std=STD), # BreakHis normalization
            ]
        )

    val_transform = T.Compose(
            [
                T.Resize((256, 256), antialias=True),  # default => (460, 700)
                T.CenterCrop(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD), # BreakHis normalization
            ]
    )
    
    num_classes = num_classes_dict[args.task]
    model = network_dict[args.net](num_classes=num_classes)
    if args.custom_afs:
        activation_function = activation_dict[args.activation]
        activation_function.replace_activation_function(model)
    # summary(model, 
    #         (3, 224, 224), 
    #         batch_dim = 0, 
    #         col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), 
    #         verbose = 1)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_dataset = datasets.BreaKHis(args.task, 'train', magnification = args.mag, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = datasets.BreaKHis(args.task, 'val', magnification = args.mag, transform=val_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dataset = datasets.BreaKHis(args.task, 'test', magnification = args.mag, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.eval:
        start_training = time.time()
        history = do_train(model, train_loader, criterion, optimizer, args.epoch, args.output_dir, args.best_metric,
                scheduler = scheduler,
                val_loader= valid_loader, 
                ckpt = args.ckpt,
                resume = True)
        end_training = time.time()
        print('Time elapsed: %.5f s' % (end_training-start_training))
    ckpt_path = os.path.join(args.output_dir, 'ckpt', 'best.pth')
    if args.ckpt is not None:
        ckpt_path = args.ckpt

    print('Testing...')
    start_testing = time.time()
    loss, _, _, _, metrics = do_eval(model, test_loader, ckpt_path=ckpt_path)
    end_testing = time.time()
    
    runtime_header = 'output_dir, training_time, inference_time\n'
    with open('artifact/test_runtime.csv', 'a') as f:
        # Check if the file is empty
        if os.stat('artifact/test_runtime.csv').st_size == 0:
            # Write the header
            f.write(runtime_header)
        f.write(f'{args.output_dir}, {end_training-start_training}, {end_testing-start_testing}\n')
        
    with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
        f.write('results on test set:\n')
        f.write(f'loss: {loss}\n')
        f.write(f'accuracy: macro: {metrics["acc"]["macro"]}, micro: {metrics["acc"]["micro"]}\n')
        f.write(f'precision: macro: {metrics["precision"]["macro"]}, micro: {metrics["precision"]["micro"]}\n')
        f.write(f'recall: macro: {metrics["recall"]["macro"]}, micro: {metrics["recall"]["micro"]}\n')
        f.write(f'f1: macro: {metrics["f1"]["macro"]}, micro: {metrics["f1"]["micro"]}\n')
        f.write(f'auroc: {metrics["auroc"]}\n')
        f.write(f'confusion matrix:\n')
        f.write(f'{metrics["confusion_matrix"]}\n')
        
    epochs = range(1, len(history['train_loss_savings']) + 1)
    history['epoch'] = epochs; df = pd.DataFrame(history); df.to_csv(f"{args.output_dir}/log_training_eval.csv", index=False)
    # Plot confusion matrix
    plot_confusion_matrix(INV_LABEL_DICTS[args.task], metrics["confusion_matrix"],  str(args.output_dir).split("/")[-1], args.output_dir)
    # Plot accuracy
    plot_acc_loss_curve(epochs, history, str(args.output_dir).split("/")[-1], args.output_dir)
    
    result_header = 'output_dir, model_name, created_at, acc_macro, acc_micro, prec_macro, prec_micro, rec_macro, rec_micro, f1_macro, f1_micro, auroc\n'
    with open('artifact/result.csv', 'a') as f:
        # Check if the file is empty
        if os.stat('artifact/result.csv').st_size == 0:
            # Write the header
            f.write(result_header)
            
        f.write(f'{args.output_dir},' +
                f'{str(args.output_dir).split("/")[-1]},' +
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},' +
                f'{metrics["acc"]["macro"]}, {metrics["acc"]["micro"]}, ' +
                f'{metrics["precision"]["macro"]}, {metrics["precision"]["micro"]}, ' +
                f'{metrics["recall"]["macro"]}, {metrics["recall"]["micro"]}, ' +
                f'{metrics["f1"]["macro"]}, {metrics["f1"]["micro"]}, ' +
                f'{metrics["auroc"]}\n'
                )
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(f'task: {args.task}\n')
        f.write(f'net: {args.net}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'epoch: {args.epoch}\n')
        f.write(f'lr: {args.lr}\n')
        f.write(f'mag: {args.mag if args.mag is not None else "All"}\n')
        if args.ckpt is not None:
            f.write(f'ckpt: {args.ckpt}\n')
        f.write(f'best_metric: {args.best_metric}\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task type')
    parser.add_argument('--net', type=str, required=True, help='network class')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--custom_afs', action='store_true', help='use custom activation funtion or not')
    parser.add_argument('--activation', type=str, help='activation function')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizization algorithm')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--mag', type=int, default=None, help='magnification')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--eval', action='store_true', help='evaluate only')
    parser.add_argument('--best_metric', type=str, default='auroc', help='metric to determine best ckpt')
    
    args = parser.parse_args()
    main(args)
