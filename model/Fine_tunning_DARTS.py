import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np
class FineTuningDARTSTrain:
    def __init__(self, label, art_classes, input_image_shape):
        self.art_classes= art_classes

        input_image= Input(shape=input_image_shape, name= "InputImg")

        #block1----
        
        conv2d_1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", name="Conv2D_1")(input_image)
        batch_norm_1 = layers.BatchNormalization(name="Batch_Normalization_1")(conv2d_1)
        relu_1 = layers.Activation(activation="relu", name="ReLU_1")(batch_norm_1)

        #block2----
        conv2d_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", name="Conv2D_2")(relu_1)
        batch_norm_2 = layers.BatchNormalization(name="Batch_Normalization_2")(conv2d_2)
        relu_2 = layers.Activation(activation="relu", name="ReLU_2")(batch_norm_2)

        #block3----
        conv2d_3 = layers.Conv2D(filters=32, kernel_size=3, padding="same", name="Conv2D_3")(relu_2)
        batch_norm_3 = layers.BatchNormalization(name="Batch_Normalization_3")(conv2d_3)

        attention_1 = layers.Attention(name="Attention_1")([batch_norm_2, batch_norm_3])
        attention_2 = layers.Attention(name="Attention_2")([batch_norm_3, attention_1])
        attention_3 = layers.Attention(name="Attention_3")([attention_1, attention_2])
        attention_4 = layers.Attention(name="Attention_4")([attention_2, attention_3])
        
        #block4----
        pooling = layers.MaxPooling2D(pool_size=(3,3), name="MaxPooling2D")(attention_4)
        flatten = layers.Flatten(name="Flatten")(pooling)
        dense_1 = layers.Dense(64, activation="relu", name="Dense_1")(flatten)

        output_layer = layers.Dense(art_classes, activation="softmax", name="Output_Layer")(dense_1) 

        model = tf.keras.Model(inputs=[input_image], outputs=[output_layer], name= "FineTuningDARTS")
        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        print("Trainable paramaters: "+str(trainable_params))

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
             metrics=["categorical_accuracy"])
                      
        checkpoint = ModelCheckpoint("./weights/"+label+"_best_weights.h5", monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        checkpoin2 = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.1, min_lr=0.000001)
        self.cbks = [checkpoint,checkpoin2]
        self.model = model