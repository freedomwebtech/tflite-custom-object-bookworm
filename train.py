import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)



train_data = object_detector.DataLoader.from_pascal_voc(
    'freedomtech/train',
    'freedomtech/train',
    ['esp8266', 'pico']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'freedomtech/validate',
    'freedomtech/validate',
    ['esp8266', 'pico']
)





spec = model_spec.get('efficientdet_lite0')




model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)



model.evaluate(val_data)


model.export(export_dir='.', tflite_filename='best.tflite')
