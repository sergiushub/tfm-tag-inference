
# coding: utf-8

# In[19]:


import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import os.path
import re
import sys
import tarfile
from subprocess import Popen, PIPE, STDOUT

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def run(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    return p.stdout.read()


# In[2]:


# All the constants to run this notebook.

model_dir = '/tmp/imagenet'
image_file = ""
num_top_predictions = 5
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

IMG_URL = 'hdfs://localhost:9000/data/im10*.jpg'
TAG_URL = 'hdfs://localhost:9000/data/meta/tags/tags10*.txt'

NUM_FEATURES = 1000
NUM_MIN_OBSERVATIONS = 3


# In[3]:


# list of most used tags ordered from the most to the less used
most_used_tags = sc.binaryFiles(TAG_URL).flatMap(lambda x:x[1].splitlines()).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).sortBy(lambda x:x[1],ascending=False).take(NUM_FEATURES)

def clean_img_rdd(x):
    key = os.path.basename(x[0]).split('.')[0][2:]    
    return (key,x[1])

def clean_tags_rdd(x):
    key = os.path.basename(x[0]).split('.')[0][4:]  
    tags = x[1].splitlines()
    value = list()
    for tag in tags:
        for pos, val in enumerate(most_used_tags):
            if val[0] == tag:
                value.append(pos)    
                break
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

maybe_download_and_extract()


# In[5]:


image_data = read_file_index()


# In[6]:


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

node_lookup_bc = sc.broadcast(node_lookup)


# In[7]:


model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')
with gfile.FastGFile(model_path, 'rb') as f:
    model_data = f.read()
    
model_data_bc = sc.broadcast(model_data)


# In[8]:


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


# In[9]:


# filter the images without tags -> x[1][1] are tags
# apply inference in images
inference_images = image_data.filter(lambda x: x[1][1]).map(apply_inference)


# In[11]:


ids = []
values = []
names = []
for id,val in enumerate(most_used_tags):
    ids.append(id)
    values.append(val[1])
    names.append(val[0])
    
import matplotlib.pyplot as plt
plt.plot(ids, values, 'g')
plt.show()


# In[12]:


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

#def merge_inference_as_label_v2(categories_and_tags):
#    tags = categories_and_tags[0]
#    categories = categories_and_tags[1]
#    paired = []
#    values = [0] * NUM_FEATURES
#    for category in categories:
#        category_id = category[0]
#        category_prob = category[1]
#        for tag in tags:
#            values[tag]=category_prob
#        paired.append((category_id,values))
#    return paired


# In[13]:


categories_to_train = inference_images.flatMap(lambda x:x[1]).map(lambda x:(x[0],1)).reduceByKey(lambda x,y:x+y).filter(lambda x:x[1]>NUM_MIN_OBSERVATIONS).sortBy(lambda x:x[0]).map(lambda x:x[0]).collect()

def merge_inference_as_label_v4(categories_and_tags):
    tags = categories_and_tags[0]
    categories = categories_and_tags[1]
    paired = []
    for category in categories:
        category_id = category[0]
        category_prob = category[1]
        paired.append((category_id,SparseVector(NUM_FEATURES, sorted(tags), [category_prob]*len(tags))))
    return paired

observation_data = inference_images.flatMap(merge_inference_as_label_v4)

#observation_data.collect()


# In[15]:


categories_to_train = [330,364]


# In[79]:


def merge_inference_as_labeledpoint(observation,category_target):
    if observation[0] == category_target:
        return LabeledPoint(1,observation[1])
    else:
        return LabeledPoint(0,observation[1])

def train_randomforest_model(dataset):
    
    model = RandomForest.trainClassifier(dataset, 2, {}, 3, seed=42)
    
    return model


def score(model,test_data):
    predictions = model.predict(test_data.map(lambda x: x.features))
    lables = test_data.map(lambda x: x.label)
    labels_and_preds= predictions.zip(lables)
    metrics = BinaryClassificationMetrics(labels_and_preds)
    
    return (metrics.areaUnderPR,metrics.areaUnderROC)

models = []

for category_target in categories_to_train:
    print("Training category {} ({})".format(node_lookup[category_target],category_target))
    observation_data_labeled = observation_data.map(lambda x: merge_inference_as_labeledpoint(x,category_target))
    train_data,test_data = observation_data_labeled.randomSplit([0.7,0.3])
    model = train_randomforest_model(train_data)
    models.append((category_target,model,score(model,test_data)))
    


# In[80]:


models


# In[76]:


lis = [(1.0,1.0),(1.0,1.0),(1.0,1.0),(1.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(1.0,1.0),(1.0,1.0),(1.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)]
res = sc.parallelize(lis)
hola = BinaryClassificationMetrics(res)
hola.areaUnderPR

hola.areaUnderROC


# # Old approach:
# 

# In[ ]:


# in this case I 
# creamos un vector de features con cada categoria como label
def merge_inference_as_label_v3(categories_and_tags):
    tags = categories_and_tags[0]
    categories = categories_and_tags[1]
    paired = []
    for category in categories:
        category_id = category[0]
        category_prob = category[1]
        paired.append((category_id,[SparseVector(NUM_FEATURES, sorted(tags), [category_prob]*len(tags))])) 
    return paired


# merge images and tags
inference_as_label_dataset = inference_images.flatMap(merge_inference_as_label_v3).reduceByKey(lambda x,y: x+y)


# In[47]:


# merge images and tags
# filters inferences with more than 50% of probability
inference_as_label = inference_images.flatMap(merge_inference_as_label).filter(lambda x:x[1][1]>0.5).aggregateByKey(list(),(lambda x,y: x+list((y,))),(lambda i,j: i+j)).filter(lambda x:len(x[1])>1).collect()

inference_as_label


# In[21]:


tag_as_label = inference_images.flatMap(merge_tag_as_label).filter(lambda x:x[1][1]>0.2).aggregateByKey(list(),(lambda x,y: x+list((y,))),(lambda i,j: i+j)).filter(lambda x:len(x[1])>1).collect()

tag_as_label

