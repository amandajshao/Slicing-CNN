import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict


def config_mpl():
    mpl.rc('lines', linewidth=1.5)
    mpl.rc('font', family='Times New Roman', size=16, monospace='Courier New')
    mpl.rc('legend', fontsize='small', fancybox=False,
           labelspacing=0.1, borderpad=0.1, borderaxespad=0.2)
    mpl.rc('figure', figsize=(12, 10))
    mpl.rc('savefig', dpi=120)


def _parse_int(content, patt_str):
    pattern = re.compile(patt_str)
    ret = pattern.findall(content)
    ret = map(float, ret)
    return ret


def _parse_float(content, patt_str):
    pattern = re.compile(patt_str)
    ret = pattern.findall(content)
    ret = [r[0] for r in ret]
    ret = map(float, ret)
    return ret


def _parse_kv(content, patt_str):
    pattern = re.compile(patt_str)
    matched = pattern.findall(content)
    ret = defaultdict(list)
    for groups in matched:
        k, v = groups[0], groups[1]
        ret[k].append(float(v))
    return ret


def parse_train(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    iter_list = _parse_int(content, r'Iteration (\d+), loss = ')
    loss_list = _parse_float(content, r'Iteration \d+, loss = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)')
    output_kv = _parse_kv(content, r'Train net output #\d+: (.+?) = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)')
    assert len(iter_list) == len(loss_list)
    for k in output_kv:
        # The new version of the caffe log computes a final loss without train
        # net output when training ends.
        if len(iter_list) == len(output_kv[k]) + 1:
            del iter_list[-1]
            del loss_list[-1]
        assert len(iter_list) == len(output_kv[k])
    return iter_list, loss_list, output_kv


def parse_test(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    iter_list = _parse_int(content, r'Iteration (\d+), Testing net')
    output_kv = _parse_kv(content, r'Test net output #\d+: (.+?) = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)')
    for k in output_kv:
        assert len(iter_list) == len(output_kv[k])
    return iter_list, output_kv


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Show Caffe training log. By default the averaged training"
                    "loss, test loss, and test accuracies will be displayed."
                    "Other train net outputs and test net outputs will also be"
                    "displayed if their names are provided in the arguments.")
    parser.add_argument('log', help="Caffe training log")
    parser.add_argument('-o', '--output', help="Save as image if necessary")
    parser.add_argument('-train', '--train-output', nargs='*', default=[],
        help="Names of train net outputs to be shown")
    parser.add_argument('-test', '--test-output', nargs='*', default=[],
        help="Names of test net outputs to be shown.")
    args = parser.parse_args()

    train_iter_list, train_loss_list, train_output_kv = parse_train(args.log)
    test_iter_list, test_output_kv = parse_test(args.log)

    config_mpl()
    legend_font = mpl.font_manager.FontProperties(family='monospace')
    color_list = ['#FF4136', '#0074D9', '#FF851B', '#7FDBFF',
                  '#F012BE', '#39CCCC', '#001F3F', '#2ECC40']

    fig = plt.figure()
    title = os.path.splitext(os.path.basename(args.log))[0]
    title = title.replace('_', ' ').title()
    plt.suptitle(title, fontsize=24, fontweight='bold')

    # determine the number of subplots
    n_rows = 1
    if 'accuracy' in test_output_kv:
        n_rows += 1
    if len(args.train_output) > 0 or len(args.test_output) > 0:
        n_rows += 1

    # plot loss
    ax = fig.add_subplot(n_rows, 1, 1)
    legend_list = []
    if 'loss' in test_output_kv:
        ax.plot(test_iter_list, test_output_kv['loss'], color=color_list[0])
        legend_list.append('test loss')
    ax.plot(train_iter_list, train_loss_list, color=color_list[len(legend_list)])
    legend_list.append('train loss')
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.legend(legend_list, loc='upper right', prop=legend_font)
    cur_row = 1

    # plot accuracy
    if 'accuracy' in test_output_kv:
        cur_row += 1
        ax = fig.add_subplot(n_rows, 1, cur_row)
        ax.plot(test_iter_list, test_output_kv['accuracy'], color=color_list[0])
        ax.set_xlabel('Iterations', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.legend(['test accuracy'], loc='lower right', prop=legend_font)

    # plot output
    if len(args.train_output) > 0 or len(args.test_output) > 0:
        cur_row += 1
        ax = fig.add_subplot(n_rows, 1, cur_row)
        legend_list = []
        for k in args.train_output:
            ax.plot(train_iter_list, train_output_kv[k], color=color_list[len(legend_list)])
            legend_list.append('train ' + k)
        for k in args.test_output:
            ax.plot(test_iter_list, test_output_kv[k], color=color_list[len(legend_list)])
            legend_list.append('test ' + k)
        ax.set_xlabel('Iterations', fontweight='bold')
        ax.set_ylabel('Output', fontweight='bold')
        ax.legend(legend_list, loc='lower right', prop=legend_font)

    if args.output is not None:
        fig.savefig(args.output)
    plt.show()
