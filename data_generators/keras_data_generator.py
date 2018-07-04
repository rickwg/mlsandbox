import threading
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class ThreadSafeIter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """
    Decorator that takes a generator function and makes it thread-safe.
    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


@thread_safe_generator
def generator_from_df(data, batch_size, target_size, rescale, features=None, debug=False):
    """
    Generator that yields (X, Y).

    """
    num_batches, n_skipped_per_epoch = divmod(data.shape[0], batch_size)
    if debug:
        print(f'Data Generator: batch size: {batch_size}, number of batch: {num_batches}, '
              f'shape of dataframe: {str(data.shape)}')

    count = 1
    epoch = 0
    while 1:
        # Shuffle each epoch. frac=1 is same as shuffling data.
        data = data.sample(frac=1)
        epoch += 1
        i, j = 0, batch_size
        mini_batches_completed = 0
        for _ in range(num_batches):

            if debug:
                print("Top of generator for loop, epoch / count / i / j = "\
                      "%d / %d / %d / %d" % (epoch, count, i, j))
            sub_data = data.iloc[i:j]
            try:
                X = np.array([rescale * img_to_array(load_img(f,
                                                              target_size=target_size,
                                                              interpolation='lanczos'))
                              for f in sub_data.imgpath])
                Y = sub_data.target.values
                if features is None:
                    # Simple model, one input, one output.
                    mini_batches_completed += 1
                    yield X, Y

                else:
                    pass
                    # Using additional features.
                    # Maybe something like:
                    # X2 = sub_data.features.values
                    # mini_batches_completed += 1
                    # yield [X, X2], Y

            except IOError as err:
                # ToDo: Error handling
                # At the moment we skip a mini-batch if some images are broken or not readable images
                count -= 1

            i = j
            j += batch_size
            count += 1
