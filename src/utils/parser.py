import argparse

def get_model_parser():
    """ model arguments for argparse """
    model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')
    model_parser.add_argument('--path', type=str, required=True, help='path to experiment')
    model_parser.add_argument('--transformer', type=str, default='roberta-large', help='transformer to use (default=robert-large)')
    model_parser.add_argument('--prompt-finetuning', action='store_true', help='whether to use prompt finetuning')
    model_parser.add_argument('--label-words', type=str, nargs='+', default=['bad', 'good'], help='which words to use as labels fro prompt finetuning (default=bad good)')
    model_parser.add_argument('--template', type=str, default='<t>', help='which words to use as labels fro prompt finetuning (default=bad good)')

    model_parser.add_argument('--loss', type=str, default=None, help='loss function to use (default = None (cross entropy))')
    model_parser.add_argument('--maxlen', type=int, default=512, help='max length of transformer inputs')
    model_parser.add_argument('--num-classes', type=int, default=2, help='number of classes (3 for NLI)')
    model_parser.add_argument('--rand-seed', type=int, default=None, help='random seed for reproducibility')
    return model_parser

def get_train_parser():
    """ training arguments for argparse """
    train_parser = argparse.ArgumentParser(description='Arguments for training the system')
    train_parser.add_argument('--dataset', type=str, default='sst', help='dataset to train the system on')
    train_parser.add_argument('--bias', type=str, default=None, help='whether data should be synthetically biased (e.g. lexical)')
    train_parser.add_argument('--lim', type=int, default=None, help='size of data subset to use for debugging')

    train_parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train system for')
    train_parser.add_argument('--bsz', type=int, default=4, help='training batch size')
    train_parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    train_parser.add_argument('--data-ordering', action='store_true', help='dynamically batches to minimize padding')

    train_parser.add_argument('--grad-clip', type=float, default=1, help='gradient clipping')
    train_parser.add_argument('--freeze-trans', type=str, default=None, help='number of epochs to freeze transformer')

    train_parser.add_argument('--log-every', type=int, default=400, help='logging training metrics every number of examples')
    train_parser.add_argument('--val-every', type=int, default=50_000, help='when validation should be done within epoch')
    train_parser.add_argument('--early-stop', type=int, default=3, help='logging training metrics every number of examples')

    train_parser.add_argument('--wandb', action='store_true', help='if set, will log to wandb')
    train_parser.add_argument('--device', type=str, default='cuda', help='selecting device to use')
    return train_parser


    