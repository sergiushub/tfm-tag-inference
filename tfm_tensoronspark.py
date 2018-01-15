
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import os.path
import re
import sys
import tarfile
from subprocess import Popen, PIPE, STDOUT

def run(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    return p.stdout.read()


# In[2]:


# All the constants to run this notebook.

model_dir = '/tmp/imagenet'
image_file = ""
num_top_predictions = 5
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

IMG_URL = 'hdfs://localhost:9000/data/im*.jpg'
TAG_URL = 'hdfs://localhost:9000/data/tags/tags*.txt'


# In[3]:


def clean_img_rdd(x):
    key = os.path.basename(x[0]).split('.')[0][2:]    
    return (key,x[1])

def clean_tags_rdd(x):
    key = os.path.basename(x[0]).split('.')[0][4:]  
    value = x[1].splitlines()
    return (key,value)
    
def read_file_index():
    im = sc.binaryFiles(IMG_URL).map(clean_img_rdd)
    tg = sc.binaryFiles(TAG_URL).map(clean_tags_rdd)
    
    return im.join(tg)


# In[4]:


def maybe_download_and_extract():
    """Download and extract model tar file."""
    from six.moves import urllib
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        filepath2, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        print("filepath2", filepath2)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('Data already downloaded:', filepath, os.stat(filepath))


# In[5]:


maybe_download_and_extract()


# In[6]:


image_data = read_file_index()


# In[7]:


label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')

def load_lookup():
    """Loads a human readable English name for each softmax node.
    
    Args:
        label_lookup_path: string UID to integer node ID.
        uid_lookup_path: string UID to human-readable string.

    Returns:
        dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
        tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
        tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
        parsed_items = p.findall(line)
        uid = parsed_items[0]
        human_string = parsed_items[2]
        uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1]
            node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
        if val not in uid_to_human:
            tf.logging.fatal('Failed to locate: %s', val)
        name = uid_to_human[val]
        node_id_to_name[key] = name

    return node_id_to_name

node_lookup = load_lookup()


# In[8]:


node_lookup_bc = sc.broadcast(node_lookup)


# In[9]:


model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')
with gfile.FastGFile(model_path, 'rb') as f:
    model_data = f.read()


# In[10]:


model_data_bc = sc.broadcast(model_data)


# In[11]:


def run_image(sess, img_id, image, tags, node_lookup):

    scores = []
    
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                            {'DecodeJpeg/contents:0': image})
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    scores = []
    for node_id in top_k:
        if node_id not in node_lookup:
            human_string = ''
        else:
            human_string = node_lookup[node_id]
        score = predictions[node_id]
        #scores.append((human_string, score))
        scores.append((node_id, score))
    return (tags, scores, img_id)

def apply_inference(image_entry):
    img_id = image_entry[0]
    image = image_entry[1][0]
    tags = image_entry[1][1]
    with tf.Graph().as_default() as g:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_data_bc.value)
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            labelled = run_image(sess, img_id, image, tags, node_lookup_bc.value)
            return labelled


# In[12]:


# filter the images without tags -> x[1][1] are tags
inference_images = image_data.filter(lambda x: x[1][1]).map(apply_inference)


# In[13]:


local_inference_images = inference_images.collect()

local_inference_images


# In[15]:


def merge_tag_as_label(categories_and_tags):
    tags = categories_and_tags[0]
    categories = categories_and_tags[1]
    paired = []
    for tag in tags:
        for category in categories:
            paired.append((tag,category))
    return paired

def merge_inference_as_label(categories_and_tags):
    tags = categories_and_tags[0]
    categories = categories_and_tags[1]
    paired = []
    for category in categories:
        for tag in tags:
            paired.append((category[0],(tag,category[1])))
    return paired


#tag_as_label = inference_images.flatMap(merge_tag_as_label).aggregateByKey((),(lambda x,y: x+(y,)),(lambda x,y: x+(y,))).collect()
#inference_as_label = inference_images.flatMap(merge_inference_as_label).aggregateByKey((),(lambda x,y: x+(y,)),(lambda x,y: x+(y,))).collect()


# In[20]:


inference_as_label = inference_images.flatMap(merge_inference_as_label).filter(lambda x:x[1][1]>0.2).aggregateByKey(list(),(lambda x,y: x+list((y,))),(lambda i,j: i+j)).filter(lambda x:len(x[1])>1).collect()

inference_as_label


# In[21]:


tag_as_label = inference_images.flatMap(merge_tag_as_label).filter(lambda x:x[1][1]>0.2).aggregateByKey(list(),(lambda x,y: x+list((y,))),(lambda i,j: i+j)).filter(lambda x:len(x[1])>1).collect()

tag_as_label


# In[28]:


tg = sc.binaryFiles(TAG_URL).flatMap(lambda x:x[1].splitlines()).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).filter(lambda x:x[1]>2)
tg.collect()

