import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import numpy as np
class VGG16:
    def __init__(self, label, art_classes, input_image_shape):
        self.art_classes= art_classes

        input_image= Input(shape=input_image_shape, name= "InputImg")

        #block1----
        VGG16 = tf.keras.applications.vgg16.VGG16(include_top= False,weights ="imagenet", input_tensor= input_image)
        for layer in VGG16.layers:
            layer.trainable = True
        flat = Flatten()(VGG16.layers[-1].output)
        fc1= Dense(units=4096,activation= "relu",name="fc1")(flat)
        fc2= Dense(units=4096,activation= "relu",name="fc2")(fc1)
        output_layer = Dense(units=art_classes,activation= "softmax", name="output_layer")(fc2)

        model = Model(inputs=[input_image], outputs=[output_layer], name= "Custom_VGG16")
        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        print("Trainable paramaters: "+str(trainable_params))
        
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss="categorical_crossentropy",
                      metrics=['categorical_accuracy'])
                      
        checkpoint = ModelCheckpoint("./weights/"+label+"_best_weights.h5", monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        checkpoin2 = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.1, min_lr=0.000001)
        self.cbks = [checkpoint,checkpoin2]
        self.model = model
        