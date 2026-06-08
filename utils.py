
import pickle
import numpy as np

def load_batch(filename):

    with open(filename, 'rb') as f:

        data = pickle.load(
            f,
            encoding='bytes'
        )

    images = data[b'data']
    labels = data[b'labels']

    images = images.reshape(
        -1,
        3,
        32,
        32
    ).transpose(
        0,
        2,
        3,
        1
    )

    return images, np.array(labels)
