
import os
import click
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.transform import resize

from keras import utils as np_utils
from keras.models import Model
from keras import models
from keras.layers import Input, Dense,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

#TODO:choose best image size to reshape
IMG_WIDTH = 299
IMG_HEIGHT= 299
EPOCHS=1
BATCH_SIZE=32

DATA_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(DATA_PATH, '../yelp_photos', 'yelp_academic_dataset_photo.json')
MODEL_PATH = DATA_PATH + "/project_best_model.h5"

#TODO: find best
def get_data(df_in):
    '''
    :param df_in: dataframe, two columns: lable, photo_id
    :return: x: image data, y: decoded labels
    '''
    labels = np.array(df_in.iloc[:, :1])
    img_id = np.array(df_in.iloc[:, 1:])
    img_id = np.ravel(img_id.reshape((1, len(img_id))))

    le = preprocessing.LabelEncoder()
    label_list = ['food', 'inside', 'outside', 'drink', 'menu']
    le.fit(label_list)
    label_to_integer = le.transform(labels)
    y = np_utils.to_categorical(label_to_integer)

    images = []
    for i in img_id:
        img_path = os.path.join(DATA_PATH, '../yelp_photos', 'yelp_academic_dataset_photos', i) +'.jpg'
        image = imread(img_path)
        image_resized = resize(image, (IMG_HEIGHT, IMG_WIDTH))
        images.append(image_resized)

    x = np.array(images)
    return x, y

def build_model():
    '''
    based on InceptionV3, initial weight for imagenet
    add global average pooling layer and fully-connected layer for prediction, output: 5
    :return: complied model
    '''

    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    inception_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    avg_glob_pool = GlobalAveragePooling2D(name='avg_glob_pool')(inception_model.output)
    fc_prediction = Dense(5, activation='softmax', name='fc_prediction')(avg_glob_pool)
    new_model = Model(inputs=inception_model.input, outputs=fc_prediction)
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    new_model.summary()
    return new_model

def call_back_list(weight_path):

    history = History()
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto', save_weights_only=True)

    # REDUCE_LR_CALLBACK = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                        patience=REDUCE_LR_PATIENCE, verbose=1, mode='min',
    #                                        min_delta=0.0001, cooldown=2, min_lr=1e-7)

    # EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
    #                                         patience=EARLY_STOPPING_PATIENCE)

    callbacks_list = [history, checkpoint]#, REDUCE_LR_CALLBACK, EARLY_STOPPING_CALLBACK]
    return callbacks_list

#TODO: modify parameters in data_generator_dict
##Build an image data generator
def img_gen(train_x, train_y, valid_x, valid_y):

    data_generator_dict = dict(featurewise_center=False,
                               samplewise_center=False,
                               rotation_range=45,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

    datagen = ImageDataGenerator(**data_generator_dict)
    ##generate data
    datagen.fit(train_x)

    train_generator = datagen.flow(train_x,train_y,
                         batch_size=BATCH_SIZE,
                         seed=None,
                         shuffle=True)

    valid_generator = datagen.flow(valid_x, valid_y,
                               batch_size=BATCH_SIZE,
                               seed=None,
                               shuffle=True)

    return train_generator,valid_generator

@click.command()
@click.option('--train', type=bool,
              default=False, help='Whether to train a model or not. True: train model, False(default):load pre-trained model')

def main(train):

    print('loading Data...')
    json_file = pd.read_json(TRAIN_DATA_PATH, orient='columns', lines=True)
    json_file.drop(['business_id', 'caption'], axis=1, inplace=True)
    # print(json_file.head())

    #TODO: change dataset to entire dataset instead 96 train samples and 96 valid samples
    #split dataset to train data and viladation data
    train_ids, valid_ids = train_test_split(json_file, test_size=96, train_size=96, stratify=json_file['label'])

    #generate image data
    train_x, train_y = get_data(train_ids)
    valid_x, valid_y = get_data(valid_ids)
    # print(train_y.shape)

    #augument and generate train data and validation data
    train_generator, valid_generator = img_gen(train_x, train_y, valid_x, valid_y)

    weight_path = "{}_weights.best.hdf5".format('project')
    callbacks_list = call_back_list(weight_path)

    if train==True:
        # biuld model
        print('building model...')
        model = build_model()

        print('Training model...')
        # fits the model on batches with real-time data augmentation:
        history_log = [model.fit_generator(train_generator, steps_per_epoch=len(train_x) / 32, \
                                           epochs=EPOCHS, callbacks=callbacks_list, validation_data=valid_generator,use_multiprocessing=False)]

        # history_log = [
        #     model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list,
        #               validation_data=(valid_x, valid_y))]

        print('Saving model...')
        model.load_weights(weight_path)
        model.save(MODEL_PATH)
        print('done')

    elif train==False:
        print('Loading model...')
        model = models.load_model(MODEL_PATH, custom_objects=None, compile=False)

    else:
        print('Please choose train(True) or load pre-trained model(False)')
        exit(0)

    #TODO: add accuracy, recalls, percision
    #TODO: use matplotlib show confuse matrix
    pred_val_y = model.predict_generator(valid_generator, verbose=1)
    matrix = confusion_matrix(valid_y.argmax(axis=1), pred_val_y.argmax(axis=1))
    print(matrix)

if __name__ == "__main__":
    main()



