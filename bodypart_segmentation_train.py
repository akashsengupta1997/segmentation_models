import numpy as np
import os
import cv2

from segmentation_models import FPN
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt


NUM_BODYPARTS = 31


def classlab(labels):
    """
    Function to convert HxWx1 labels image to HxWxC one hot encoded matrix.
    :param labels: HxWx1 labels image
    :return: HxWxC one hot encoded matrix.
    """
    # There are 32 classes per pixel - 0 is background, 1-31 is bodyparts
    num_classes = NUM_BODYPARTS + 1
    x = np.zeros((labels.shape[0], labels.shape[1], num_classes))
    # print('IN CLASSLAB', labels.shape)
    for pixel_class in range(num_classes):
        indexes = list(zip(*np.where(labels == pixel_class)))
        for index in indexes:
            x[index[0], index[1], pixel_class] = 1.0
    # print("class lab shape", x.shape)
    return x


def generate_data(image_generator, mask_generator, n):
    images = []
    labels = []
    i = 0
    while i < n:
        x = image_generator.next()
        y = mask_generator.next()
        # x and y are batches of images and masks
        # plt.imshow(y[0, :, :, 0]*8)
        # plt.show()
        # print(str(len(images))+ " " +str(x.shape))
        # print('x shape in generate data', x.shape) # should = (batch_size, img_hw, img_hw, 3)
        # print('y shape in generate data', y.shape) # should = (batch_size, dec_hw, dec_hw, 1)
        j = 0
        while j < x.shape[0]:
            images.append(x[j, :, :, :])
            labels.append(classlab(y[j, :, :, :].astype(np.uint8)))
            j = j + 1
            i = i + 1
            if i >= n:
                break

    # print('images shape in generate data', np.array(images).shape,
    #       'labels shape in generate data', np.array(labels).shape)
    return np.array(images), np.array(labels)


def test(train_data, model, img_wh, img_dec_wh, image_dir, save=False):
    img_list = []
    if not (train_data is None):
        for id in range(0, 10):
            # plt.imshow((train_data[0][id,:,:,:]).astype(np.uint8))
            # plt.show()
            img_list.append(train_data[0][id, :, :, :])
    # plt.imshow((img_list[id]).astype(np.uint8))
    # plt.show()

    fnames = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            print(fname)
            image = cv2.imread(os.path.join(image_dir, fname))
            image = cv2.resize(image, (img_wh, img_wh))
            image = image[..., ::-1]
            # plt.imshow(image)
            # plt.show()
            img_list.append(image / 255.0)
            fnames.append(fname)

    img_tensor = np.array(img_list)
    output = np.reshape(model.predict(img_tensor), (len(img_list), img_dec_wh, img_dec_wh,
                                                    NUM_BODYPARTS+1))
    print("orig output shape", output.shape)
    for img_num in range(len(img_list)):
        seg_labels = output[img_num, :, :, :]
        seg_img = np.argmax(seg_labels, axis=2)
        print("labels output shape", seg_labels.shape)
        print("seg img output shape", seg_img.shape)
        if not save:
            plt.figure(1)
            plt.clf()
            plt.subplot(331)
            plt.imshow(seg_labels[:, :, 0], cmap="gray")
            plt.subplot(332)
            plt.imshow(seg_labels[:, :, 3], cmap="gray")
            plt.subplot(333)
            plt.imshow(seg_labels[:, :, 6], cmap="gray")
            plt.subplot(334)
            plt.imshow(seg_labels[:, :, 9], cmap="gray")
            plt.subplot(335)
            plt.imshow(seg_labels[:, :, 12], cmap="gray")
            plt.subplot(336)
            plt.imshow(seg_labels[:, :, 15], cmap="gray")
            plt.subplot(337)
            plt.imshow(seg_labels[:, :, 18], cmap="gray")
            plt.subplot(338)
            plt.imshow(seg_labels[:, :, 21], cmap="gray")
            plt.subplot(339)
            plt.imshow(seg_labels[:, :, 24], cmap="gray")
            plt.figure(2)
            plt.clf()
            plt.imshow(seg_img*8)
            plt.figure(3)
            plt.clf()
            plt.imshow(img_list[img_num])
            plt.show()
        else:
            save_path = os.path.join(image_dir, "results", os.path.splitext(fnames[img_num])[0]
                                     + "_seg_img.png")
            plt.imsave(save_path, seg_img * 8)


def segmentation_train():
    img_wh = 256
    img_dec_wh = 64
    batch_size = 1
    test_image_dir_list = []  # TODO

    train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images"
    train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks"
    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'

    image_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale = 1/255.0,
        fill_mode='nearest')

    mask_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    image_datagen = ImageDataGenerator(**image_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)

    # Provide the same seed to flow methods
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        target_size=(img_dec_wh, img_dec_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)
    print('Generators loaded.')

    model = FPN(backbone_name='resnet50',
                encoder_weights=None,
                classes=32,
                input_shape=(img_wh, img_wh, 3),
                last_upsample=1
                )

    model.compile('Adam', 'categorical_crossentropy', ['accuracy'])
    print(model.summary())

    for trials in range(11, 4000):
        nb_epoch = 1
        print("Fitting", trials)

        def data_gen():
            while True:
                train_data, train_label_images = generate_data(image_generator, mask_generator,
                                                               batch_size)
                train_label = np.reshape(train_label_images,
                                         (batch_size, img_dec_wh * img_dec_wh,
                                          NUM_BODYPARTS + 1))
                yield (train_data, train_label)

        history = model.fit_generator(data_gen(), steps_per_epoch=int(1 / batch_size),
                                      nb_epoch=nb_epoch, verbose=1)

        print("After fitting")
        if trials % 100 == 0:
            model.save('bodypart_test_models/test_weight'
                       + str(nb_epoch * (trials + 1)).zfill(4) + '.hdf5')


def segmentation_test(img_wh, img_dec_wh, save=False):
    test_image_dir = 'test_videos/my_vid1'
    print('Preloaded model')
    autoencoder = load_model('/Users/Akash_Sengupta/Documents/GitHub/segmentation_models/'
                             'bodypart_models/FPN_resnet50_256_2101.hdf5')
    test(None, autoencoder, img_wh, img_dec_wh, test_image_dir, save=save)

# segmentation_train()
segmentation_test(256, 256, save=True)