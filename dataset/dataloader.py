
import pickle
import tensorflow as tf


def return_dataset(args, path, length=-1):
    raw_data = pickle.load(open(path, 'rb'))

    """## Create tf.data.Dataset object"""
    data_input = raw_data[:length]
    data_target = raw_data[:length]

    # 每个句子都补长到同样长度[[1,2,56,……], [64, 24, 64……],……]
    data_input = tf.keras.preprocessing.sequence.pad_sequences(data_input, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices((data_input, data_input))
    dataset = dataset.cache()  # 放在缓存里
    dataset = dataset.shuffle(args.shuffle_size).batch(args.bs)  # 打乱的batch和取数据的batch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # 预取数据
    return dataset


def return_loader(args):
    """## Load data"""
    train_dataset = return_dataset(args, args.train_save_path, -1)
    test_dataset = return_dataset(args, args.test_save_path, -1)

    return train_dataset, test_dataset

