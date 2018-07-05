import os
import argparse

import pandas as pd


def cifar10_create_file_list(cifat10_path: str) -> pd.DataFrame:
    # Iterate over data batches
    data = pd.DataFrame()
    for i, fpath in enumerate(os.scandir(cifat10_path)):
        data_dict = {'filepath': list(), 'label': list(), 'splitname': list()}
        print(f'At batch: {fpath.name}')
        # Mark train and test bathces
        if 'test' in fpath.name:
            data_split_name = 'test'
        else:
            data_split_name = 'train'
        # Iterate over labels
        for lpath in os.scandir(fpath.path):
            # Iterate over images
            for ipath in os.scandir(lpath.path):
                data_dict['filepath'].append(ipath.path)
                data_dict['label'].append(lpath.name)
                data_dict['splitname'].append(data_split_name)

        data = data.append(pd.DataFrame(data=data_dict), ignore_index=True)
    return data


def main(path):
    cifar10_path = path
    out_fname = 'cifar10DF.csv'
    df = cifar10_create_file_list(cifar10_path)
    df.to_csv(os.path.join(cifar10_path, out_fname), sep=';', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default=None,
        type=str,
        help='Data path for cifar10.',
        action='store_true')
    args = parser.parse_args()
    if args.data_path:
        main(path=args.data_path)
    else:
        print('Please, set the data path')
