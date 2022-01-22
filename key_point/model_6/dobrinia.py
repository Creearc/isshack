import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses

def dobro_module(conc, CLASSES_NUM):

  conc = depthwise_conv(conc,
                          filters=64,
                          kernel_size=5,
                          strides=1)
  
  conc = inception_module(conc,
                     filters_1x1=16,
                     filters_3x3_reduce=16,
                     filters_3x3=64,
                     filters_5x5_reduce=16,
                     filters_5x5=128,
                     filters_pool_proj=32,
                     name='inception_3a') 
  
  for i in range(2):
    conc = depthwise_nikita_layer(conc,
                                  filters_1=32,
                                  filters_2=64)

  conc = depthwise_conv(conc,
                        filters=64,
                        kernel_size=5,
                        strides=2)

  conc = layers.Dropout(0.2)(conc)

  conc = layers.GlobalAveragePooling2D()(conc)
  conc = layers.Dense(CLASSES_NUM * 100, activation='relu')(conc) 
  conc = layers.Dense(CLASSES_NUM, activation='relu')(conc)
  return conc

class Model:
  def __init__(self):   
    input_layer = layers.Input(shape=340)
    
    conc = layers.Dense(680, activation='relu')(conc)
    conc = layers.Dense(340, activation='relu')(conc) 
    conc = layers.Dense(1, activation='softmax')(conc)

    self.model = tf.keras.Model(input_layer, conc)






    
