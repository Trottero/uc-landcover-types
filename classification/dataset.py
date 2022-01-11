#%%
import os
import tifffile
import tensorflow as tf
import numpy as np

generated_folder = "../gen-cloudless/out/DSen2-CR_001/predictions/"
pristine_folder = "../gen-cloudless/data/ROIs1868_summer_s2/"
cloudy_folder = "../gen-cloudless/data/ROIs1868_summer_s2_cloudy/"

base = "ROIs1868_summer"
pristine_prefix = "s2"
cloudy_prefix = "s2_cloudy"


def get_generated_file_ids():
    # get all files in the dirs
    files = []
    for id in os.listdir(generated_folder):
        if id.endswith(".tif"):
            files.append(id)
            
    files = set(files)
    return list(files)

def get_data(n):
    generated_imgs = np.array([])
    for id in get_generated_file_ids()[:n]:
        img = tifffile.imread(os.path.join(generated_folder, id), )
        img = img[:13].transpose(2,1,0)[32:32+64, 32:32+64, :]
        generated_imgs = np.array([*generated_imgs, img], dtype=np.float32)
    # print(generated_imgs.shape)

    pristine_imgs = np.array([])
    for id in get_generated_file_ids()[:n]:
        main_id = id.split("_")[0]
        path = os.path.join(pristine_folder, f'{pristine_prefix}_{main_id}', f'{base}_{pristine_prefix}_{id}')
        img = tifffile.imread(path)[96:96+64, 96:96+64, :]
        pristine_imgs = np.array([*pristine_imgs, img], dtype=np.float32)
    # print(pristine_imgs.shape)

    cloudy_imgs = np.array([])
    for id in get_generated_file_ids()[:n]:
        main_id = id.split("_")[0]
        path = os.path.join(cloudy_folder, f'{cloudy_prefix}_{main_id}', f'{base}_{cloudy_prefix}_{id}')
        img = tifffile.imread(path)[96:96+64, 96:96+64, :]
        cloudy_imgs = np.array([*cloudy_imgs, img], dtype=np.float32)
    # print(cloudy_imgs.shape)

    generated_imgs_tf = tf.data.Dataset.from_tensor_slices(generated_imgs)
    pristine_imgs_tf = tf.data.Dataset.from_tensor_slices(pristine_imgs)
    cloudy_imgs_tf = tf.data.Dataset.from_tensor_slices(cloudy_imgs)
    return generated_imgs_tf, pristine_imgs_tf, cloudy_imgs_tf
