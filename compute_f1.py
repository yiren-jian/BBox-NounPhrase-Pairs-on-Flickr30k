import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='method2/stats.txt', type=str)
    args = parser.parse_args()

    print(args.log_file)
    with open(args.log_file) as f:
        lines = f.readlines()

    TP = 0
    FP = 0
    P = 0
    for line in lines:
        P  += int(line.split(',')[3].split(':')[-1])
        TP += int(line.split(',')[4].split(':')[-1])
        FP += int(line.split(',')[5].split(':')[-1].replace('\n', ''))

    precision = TP / (TP + FP)
    print('precision', precision)
    recall = TP / P
    print('recall', recall)

    F1 = 2 * precision * recall / (precision + recall)
    print('F1', F1)
