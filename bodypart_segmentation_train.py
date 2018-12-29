import numpy as np
import os
import cv2

from segmentation_models import FPN
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt


NUM_BODYPARTS = 31


def labels_from_seg_image(seg_image_batch):
    """
    PPP seg_images have pixels labelled as follows:
    Lower leg - 89
    Upper leg - 52
    Lower arm - 14
    Upper arm - 113
    Torso - 75
    Head - 38
    Background - 0

    This function changes labels to:
    Lower leg - 1
    Upper leg - 2
    Lower arm - 3
    Upper arm - 4
    Torso - 5
    Head - 6
    Background - 0

    :param seg_image_batch: batch of segmentation masks with Pascal VOC labels
    :return: relabelled batch of segmentation masks
    """
    copy_batch = seg_image_batch.copy()
    copy_batch[seg_image_batch == 89] = 1
    copy_batch[seg_image_batch == 52] = 2
    copy_batch[seg_image_batch == 14] = 3
    copy_batch[seg_image_batch == 113] = 4
    copy_batch[seg_image_batch == 75] = 5
    copy_batch[seg_image_batch == 38] = 6
    return copy_batch


def classlab(labels, num_classes):
    """
    Function to convert HxWx1 labels image to HxWxC one hot encoded matrix.
    :param labels: HxWx1 labels image
    :return: HxWxC one hot encoded matrix.
    """
    x = np.zeros((labels.shape[0], labels.shape[1], num_classes))
    # print('IN CLASSLAB', labels.shape)
    for pixel_class in range(num_classes):
        indexes = list(zip(*np.where(labels == pixel_class)))
        for index in indexes:
            x[index[0], index[1], pixel_class] = 1.0
    # print("class lab shape", x.shape)
    return x


def generate_data(image_generator, mask_generator, n, num_classes):
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
            labels.append(classlab(y[j, :, :, :].astype(np.uint8), num_classes))
            j = j + 1
            i = i + 1
            if i >= n:
                break

    # print('images shape in generate data', np.array(images).shape,
    #       'labels shape in generate data', np.array(labels).shape)
    return np.array(images), np.array(labels)


def test(train_data, model, img_wh, img_dec_wh, image_dir, num_classes, save=False):
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
                                                    num_classes))
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
            plt.figure(2)
            plt.clf()
            plt.imshow(seg_img)
            plt.figure(3)
            plt.clf()
            plt.imshow(img_list[img_num])
            plt.show()
        else:
            if img_dec_wh == 64:
                save_folder = 'results64'
            elif img_dec_wh == 256:
                save_folder = 'results256'
            save_path = os.path.join(image_dir, save_folder, os.path.splitext(fnames[img_num])[0]
                                     + "_seg_img.png")
            plt.imsave(save_path, seg_img * 8)


def segmentation_train(img_wh, img_dec_wh, dataset):
    batch_size = 10  # TODO change back to 10

    if dataset == 'up-s31':
        train_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/images"
        train_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/masks"
        # TODO create validation directory
        num_classes = 32
        num_train_images = 8515

    elif dataset == 'ppp':
        train_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/train_images'
        train_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/train_masks'
        val_image_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/val_images'
        val_label_dir = '/Users/Akash_Sengupta/Documents/4th_year_project_datasets/VOC2010/pascal_person_part/val_masks'
        num_classes = 7
        num_train_images = 3034
        num_val_images = 500

    assert os.path.isdir(train_image_dir), 'Invalid image directory'
    assert os.path.isdir(train_label_dir), 'Invalid label directory'
    assert os.path.isdir(val_image_dir), 'Invalid validation image directory'
    assert os.path.isdir(val_label_dir), 'Invalid validation label directory'

    train_image_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1 / 255.0,
        fill_mode='nearest')

    train_mask_data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_image_data_gen_args = dict(
        rescale=(1 / 255.0),
        fill_mode='nearest')

    val_mask_data_gen_args = dict(
        fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**train_image_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_mask_data_gen_args)
    val_image_datagen = ImageDataGenerator(**val_image_data_gen_args)
    val_mask_datagen = ImageDataGenerator(**val_mask_data_gen_args)

    # Provide the same seed to flow methods for train generators
    seed = 1
    train_image_generator = train_image_datagen.flow_from_directory(
        train_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_label_dir,
        batch_size=batch_size,
        target_size=(img_dec_wh, img_dec_wh),
        class_mode=None,
        color_mode="grayscale")

    val_image_generator = val_image_datagen.flow_from_directory(
        val_image_dir,
        batch_size=batch_size,
        target_size=(img_wh, img_wh),
        class_mode=None)

    val_mask_generator = val_mask_datagen.flow_from_directory(
        val_label_dir,
        batch_size=batch_size,
        target_size=(img_dec_wh, img_dec_wh),
        class_mode=None,
        color_mode="grayscale",
        seed=seed)

    print('Generators loaded.')

    if img_dec_wh == 256:
        last_upsample = 4
    elif img_dec_wh == 64:
        last_upsample = 1
        
    model = FPN(backbone_name='resnet50',
                encoder_weights=None,
                classes=num_classes,
                input_shape=(img_wh, img_wh, 3),
                last_upsample=last_upsample
                )

    model.compile('Adam', 'categorical_crossentropy', ['accuracy'])
    # print(model.summary())

    for trials in range(4000):
        nb_epoch = 1
        print("Fitting", trials)

        def train_data_gen():
            while True:
                train_data, train_labels = generate_data(train_image_generator,
                                                         train_mask_generator,
                                                         batch_size, num_classes)
                reshaped_train_labels = np.reshape(train_labels,
                                                   (batch_size, img_dec_wh * img_dec_wh,
                                                    num_classes))
                yield (train_data, reshaped_train_labels)

        def val_data_gen():
            while True:
                val_data, val_labels = generate_data(val_image_generator,
                                                     val_mask_generator,
                                                     batch_size, num_classes)
                reshaped_val_labels = np.reshape(val_labels,
                                                 (batch_size, img_dec_wh * img_dec_wh,
                                                  num_classes))
                yield (val_data, reshaped_val_labels)

        history = model.fit_generator(train_data_gen(),
                                      steps_per_epoch=int(num_train_images / batch_size),
                                      nb_epoch=nb_epoch,
                                      verbose=1,
                                      validation_data=val_data_gen(),
                                      validation_steps=int(
                                      num_val_images) / batch_size)

        print("After fitting")
        if trials % 200 == 0:
            model.save('overfit_tests/ppp_test_weight'
                             + str(nb_epoch * (trials + 1)).zfill(4) + '.hdf5')

def segmentation_test(img_wh, img_dec_wh, save=False):
    test_image_dir = 'test_videos/my_vid1'
    print('Preloaded model')
    autoencoder = load_model('/Users/Akash_Sengupta/Documents/GitHub/segmentation_models/'
                             'bodypart_models/FPN_resnet50_64_0601.hdf5')
    test(None, autoencoder, img_wh, img_dec_wh, test_image_dir, save=save)

segmentation_train(256, 256, 'ppp')
# segmentation_test(256, 64, save=True)