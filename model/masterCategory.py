import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import sys


class MasterCategory:
    '''This model is based off of VGG16 with the addition of BatchNorm layers and then branching '''

    def __init__(self, label, master_classes, sub_classes, art_classes, input_img_shape):
        self.master_classes=master_classes
        self.sub_classes=sub_classes
        self.art_classes=art_classes
        
        input_image = Input(shape=input_img_shape,name="InputImg")

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- block 5 masterCategory ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_mas')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_mas')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_mas')(x)
        x = BatchNormalization()(x)

        #--- masterCategory prediction---
        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='fc_mas')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2_mas')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        pred = Dense(self.master_classes, activation='softmax', name='master_output')(x)

        model = Model(
            inputs=input_image,
            outputs=pred,
            name="Baseline_masterCategory_CNN")

        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        #trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("Trainable paramaters: "+str(trainable_params))

        #Keras will automaticall use categorical accuracy when accuracy is used.
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        checkpoint = ModelCheckpoint("./weights/"+label+"_best_weights.h5", monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        self.cbks = [checkpoint]
        self.model = model