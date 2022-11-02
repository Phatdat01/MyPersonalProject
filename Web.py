from PIL import Image
import streamlit as st
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

CONFIG_PATH = 'pipeline.config'
ANNOTATION_PATH = 'annotations'

tf.keras.backend.clear_session()
model = tf.saved_model.load("saved_model")

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

def XoaTrung(a, L):
    index = []
    flag = np.zeros(L, np.bool)
    for i in range(0, L):
        if flag[i] == False:
            flag[i] = True
            x1 = (a[i,0] + a[i,2])/2
            y1 = (a[i,1] + a[i,3])/2
            for j in range(i+1, L):
                x2 = (a[j,0] + a[j,2])/2
                y2 = (a[j,1] + a[j,3])/2
                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if d < 0.2:
                    flag[j] = True
            index.append(i)
    for i in range(0, L):
        if i not in index:
            flag[i] = False
    return flag

def Recognition(imgin):
        content = Image.open(imgin)
        image_np = np.asarray(content)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis,...]

        model_fn = model.signatures['serving_default']
        detections = model_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        my_box = detections['detection_boxes']
        my_class = detections['detection_classes']
        my_score = detections['detection_scores']

        my_score = my_score[my_score >= 0.7]
        L = len(my_score)
        my_box = my_box[0:L]
        my_class = my_class[0:L]
        
        flagTrung = XoaTrung(my_box, L)
        my_box = my_box[flagTrung]
        my_class = my_class[flagTrung]
        my_score = my_score[flagTrung]

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                my_box,
                my_class,
                my_score,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.7,
                agnostic_mode=False,
                line_thickness=8)

        return image_np_with_detections

st.header("This is Anime Recognition Web")
file=st.file_uploader("Choose file",type=['jpg', 'png', 'jpeg'])
img_placeholder = st.empty()
if file is not None:
    button=st.button('Recognize fruit')
    if button:
        file=Recognition(file)
    image=st.image(file)
    st.write("File Uploaded Successfully!")
    ####################################