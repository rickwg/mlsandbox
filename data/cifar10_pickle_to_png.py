import pickle
import os
import png


def cifar10_pickle_to_png(cifar10_path):
    label_path = os.path.join(cifar10_path, 'batches.meta')
    meta_file = open(label_path, 'rb')
    labels = pickle.load(meta_file, encoding="ASCII")
    for fpath in os.scandir(cifar10_path):
        if 'data' not in fpath.name and 'test' not in fpath.name:
            continue
        f = open(fpath, 'rb')
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v

        d = d_decoded
        f.close()
        for i, filename in enumerate(d['filenames']):
            folder = os.path.join(
            cifar10_path,
            'cifar10',
            fpath.name,
            labels['label_names'][d['labels'][i]],
            )
            os.makedirs(folder, exist_ok=True)
            q = d['data'][i]
            print(filename)
            with open(os.path.join(folder, filename.decode()), 'wb') as outfile:
                png.from_array(q.reshape((32, 32, 3), order='F').swapaxes(0,1), mode='RGB').save(outfile)


cifar10_path = '/home/rick/development/data_local/cifar-10-batches-py'
cifar10_pickle_to_png(cifar10_path)
