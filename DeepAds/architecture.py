'''An implementation of the DeepAds network described in https://arxiv.org/pdf/1910.03227.pdf using Keras. This
is run on the Cityscape dataset, available here: https://www.cityscapes-dataset.com/.
'''

from .base import BASE
from data.loader import LOADER
import cv2
from utils.utils import get_callbacks
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from utils.BilinearUpSampling import *
from keras.regularizers import l2
import segmentation_models as sm

class MODEL(BASE):
    def __init__(self, config):
        model_name = 'morph'
        super(MODEL, self).__init__(config, model_name)

    def compose_model(self,pretrained_weights = None,input_size = (200,200,3),weight_decay=0):
        inp = keras.layers.Input(shape=tuple(self.config.MODEL['MODEL_PARAMS']['INPUT_SHAPE']))

        # Encoder
        x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inp)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # Decoder
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

        model = keras.models.Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

        return model