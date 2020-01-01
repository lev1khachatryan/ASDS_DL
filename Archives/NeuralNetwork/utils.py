import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.python import graph_util
import numpy as np
from shutil import rmtree


def validate(path):
    path = os.path.abspath(path) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def find_all_files(path, extensions=[], exclude=[]):
    # Find all images with given extensions
    path = os.path.abspath(path) + '/'
    if len(extensions) > 0:
        files = [path + item for item in os.listdir(path) if item.split('.')[-1].lower() in extensions]
    else:
        files = [path + item for item in os.listdir(path) if os.path.isfile(path + item)]
    dirs = [item for item in os.listdir(path) if (os.path.isdir(path + item) and item not in exclude)]
    for item in dirs:
        files += find_all_files(path + item, extensions)
    return files


def load_graph(graph_path, return_elements=[]):
    # Creates graph from saved graph_def.pb
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output_nodes = tf.import_graph_def(graph_def, return_elements=return_elements)
        return output_nodes


def load_graph2(graph_path, inp, inp_name='/data', out_name='/prob:0', graph_name='model'):
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map={inp_name: inp}, name=graph_name)
        output_nodes = tf.get_default_session().graph.get_tensor_by_name(graph_name + out_name)
        return output_nodes


def freeze_save_graph(sess, name, output_node, log_dir):
    for node in sess.graph.as_graph_def().node:
        node.device = ""
    variable_graph_def = sess.graph.as_graph_def()
    optimized_net = graph_util.convert_variables_to_constants(sess, variable_graph_def, [output_node])
    tf.train.write_graph(optimized_net, log_dir, name, False)


def next_batch(data, batch_size):
    if len(data) <= batch_size:
        return [], data
    else:
        return data[batch_size:], data[:batch_size]


def next_batch_shuffle(data, batch_size):
    """
    :param data: 1D array
    :return:
    """
    if len(data) <= batch_size:
        return [], np.random.choice(data, batch_size)
    else:
        np.random.shuffle(data)
        return data[batch_size:], data[:batch_size]


def next_batch_shuffle_nd(data, batch_size):
    if len(data) <= batch_size:
        np.random.shuffle(data)
        return [], data
    else:
        mask = np.zeros(len(data), bool)
        inds = np.random.choice(range(len(mask)), batch_size, False)
        mask[inds] = True
        return data[mask], data[~mask]


def save(sess, saver, model_path, model_name):
    with tf.name_scope('saver'):
        saver.save(sess, os.path.join(model_path, model_name))


def load(sess, saver, model_path):
    print("Reading checkpoint ...")
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        model = os.path.basename(ckpt.model_checkpoint_path)
        next_epoch = int(model.split('_')[-1]) + 1
        saver.restore(sess, os.path.join(model_path, model))
        return True, next_epoch
    else:
        return False, 0


def clear_start(paths):
    for path in paths:
        validate(path)
        rmtree(path)
        validate(path)


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        std = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        mx = tf.reduce_max(var)
        mn = tf.reduce_min(var)
    tf.summary.scalar(name + '/mean', mean)
    tf.summary.scalar(name + '/stddev', std)
    tf.summary.scalar(name + '/max', mx)
    tf.summary.scalar(name + '/min', mn)
    tf.summary.histogram(name, var)


def construct_label(i, num_classes):
    l = np.zeros(num_classes)
    l[i] = 1
    return l


