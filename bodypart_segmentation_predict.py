import os

from matplotlib import pyplot as plt
import numpy as np
import cv2

from keras.models import load_model
from segmentation_models import FPN


def pad_image(img):
    """
    Pad image with 0s to make it square
    :param image: HxWx3 numpy array
    :return: AxAx3 numpy array (square image)
    """
    height, width, _ = img.shape

    if width < height:
        border_width = (height - width) // 2
        padded = cv2.copyMakeBorder(img, 0, 0, border_width, border_width,
                                    cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        padded = cv2.copyMakeBorder(img, border_width, border_width, 0, 0,
                                    cv2.BORDER_CONSTANT, value=0)

    return padded


def load_input_img(image_dir, fname, input_wh, pad=False):
    input_img = cv2.imread(os.path.join(image_dir, fname))
    if pad:
        input_img = pad_image(input_img)
    input_img = cv2.resize(input_img, (input_wh, input_wh))
    input_img = input_img[..., ::-1]
    input_img = input_img * (1.0 / 255)
    input_img = np.expand_dims(input_img, axis=0)  # need 4D input (add batch dimension)
    return input_img


def predict(test_image_dir, model_path, input_wh, output_wh, num_classes, save=False,
            pad=True):

    seg_model = load_model(model_path)

    for fname in sorted(os.listdir(test_image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            print(fname)
            input_img = load_input_img(test_image_dir, fname, input_wh, pad=pad)

            seg = seg_model.predict(input_img)
            seg = np.reshape(seg, (1, output_wh, output_wh, num_classes))
            seg_img = np.argmax(seg[0], axis=-1)

            if save:
                save_path = os.path.join(test_image_dir, "fpn_segs", fname)
                cv2.imwrite(save_path, seg_img)


predict("/data/cvfs/as2562/4th_year_proj_datasets/s31_padded_small_glob_rot/val_images/val/",
        "./up-s31_body_part_models/fpn256_small_glob_rot_no_horiz_flip_0201.hdf5",
        256,
        256,
        32,
        save=True,
        pad=False)