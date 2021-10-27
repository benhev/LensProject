from tensorflow.keras.models import Model, load_model
from LensCNN import npy_read, npy_get_shape, dir_menu, get_dir, get_nat, get_training_files
from os.path import isdir, basename
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


# TODO: fix image sizes and title placement or revert
# TODO use bookmarked site with block formations

def feature_map(model_dir: str = '', data_dir: str = ''):
    if not model_dir:
        model_dir = dir_menu(pattern='models/*/', prompt='Choose model to load')
    model_name = dir_menu(pattern=f'{model_dir}/*/*.h5', prompt='Choose checkpoint to load', sanitize=model_dir)
    model = load_model(f'{model_dir}/{model_name}')
    Path(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}').mkdir(parents=True, exist_ok=True)

    layer_names = [layer.name.replace('2d', '').replace('_', ' ').title() for layer in
                   model.layers if not layer.name == 'conv2d_5']
    layer_outputs = [layer.output for layer in model.layers if not layer.name == 'conv2d_5']

    # layer_names = layer_names[:-1]
    # layer_outputs = layer_outputs[:-1]

    feature_map_model = Model(inputs=model.input, outputs=layer_outputs)
    if not isdir(data_dir):
        print('No valid data directory supplied.')
        data_dir = get_dir('data', new=False)
    data_files = get_training_files(training_dir=data_dir, validation=False)
    xfile = data_files.get('training_input')
    yfile = data_files.get('training_label')

    max_ind = npy_get_shape(xfile)[0]
    ind = get_nat('image index', max_ind)
    x = npy_read(xfile, start_row=ind, num_rows=1)
    y = npy_read(yfile, start_row=ind, num_rows=1)
    feature_maps = feature_map_model.predict(x)
    prediction = model.predict(x)
    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
    fig1.subplots_adjust(bottom=0.05, top=0.8)
    fig1.suptitle('Final prediction compared to ground truth')
    ax1[0].title.set_text('Input Image')
    ax1[0].imshow(x.reshape(x.shape[1:]))
    ax1[1].title.set_text('Prediction')
    ax1[1].imshow(prediction.reshape(prediction.shape[1:]))
    ax1[2].title.set_text('Ground Truth')
    ax1[2].imshow(y.reshape(y.shape[1:]))
    length_per_layer = 3
    total_width = 30
    # fig, ax = plt.subplots(len(layer_names), 1, figsize=(total_width, len(layer_names) * length_per_layer))
    # fig.suptitle(f'Feature maps for model {model.name} at checkpoint {basename(model_name)}')
    # fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, hspace=0.3)
    for i, (layer_name, feat_map) in enumerate(zip(layer_names, feature_maps)):
        if len(feat_map.shape) == 4:
            k = feat_map.shape[-1]
            size = feat_map.shape[1]
            image_belt = np.zeros((size, k * size))
            for j in range(k):
                feature_image = feat_map[0, :, :, j]
                # feature_image -= feature_image.mean()
                # feature_image /= feature_image.std()
                # feature_image *= 64
                # feature_image += 128
                # feature_image = np.clip(x, 0, 255).astype('uint8')
                image_belt[:, j * size: (j + 1) * size] = feature_image
            scale = total_width / k
            fig = plt.figure(figsize=(scale * k, scale))
            ax = fig.add_subplot()
            ax.title.set_text(layer_name + f' ({k} images)')
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85)
            ax.axis('off')
            ax.grid(False)
            ax.imshow(image_belt, aspect='auto')
            fig.savefig(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}/{layer_name}.jpg')
            fig.clf()
    fig1.savefig(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}/prediction.jpg')
    plt.close('all')


def feature_map1(model_dir: str = '', data_dir: str = ''):
    if not model_dir:
        model_dir = dir_menu(pattern='models/*/', prompt='Choose model to load')
    model_name = dir_menu(pattern=f'{model_dir}/*/*.h5', prompt='Choose checkpoint to load', sanitize=model_dir)
    model = load_model(f'{model_dir}/{model_name}')
    Path(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}').mkdir(parents=True, exist_ok=True)

    layer_names = [layer.name.replace('2d', '').replace('_', ' ').title() for layer in
                   model.layers if not layer.name == 'conv2d_5']
    layer_outputs = [layer.output for layer in model.layers if not layer.name == 'conv2d_5']

    # layer_names = layer_names[:-1]
    # layer_outputs = layer_outputs[:-1]

    feature_map_model = Model(inputs=model.input, outputs=layer_outputs)
    if not isdir(data_dir):
        print('No valid data directory supplied.')
        data_dir = get_dir('data', new=False)
    data_files = get_training_files(training_dir=data_dir, validation=False)
    xfile = data_files.get('training_input')
    yfile = data_files.get('training_label')

    max_ind = npy_get_shape(xfile)[0]
    ind = get_nat('image index', max_ind)
    x = npy_read(xfile, start_row=ind, num_rows=1)
    y = npy_read(yfile, start_row=ind, num_rows=1)
    feature_maps = feature_map_model.predict(x)
    prediction = model.predict(x)
    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
    fig1.subplots_adjust(bottom=0.05, top=0.8)
    fig1.suptitle('Final prediction compared to ground truth')
    ax1[0].title.set_text('Input Image')
    ax1[0].imshow(x.reshape(x.shape[1:]))
    ax1[1].title.set_text('Prediction')
    ax1[1].imshow(prediction.reshape(prediction.shape[1:]))
    ax1[2].title.set_text('Ground Truth')
    ax1[2].imshow(y.reshape(y.shape[1:]))
    length_per_layer = 3
    total_width = 30
    # fig, ax = plt.subplots(len(layer_names), 1, figsize=(total_width, len(layer_names) * length_per_layer))
    # fig.suptitle(f'Feature maps for model {model.name} at checkpoint {basename(model_name)}')
    # fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, hspace=0.3)
    for i, (layer_name, feat_map) in enumerate(zip(layer_names, feature_maps)):
        if len(feat_map.shape) == 4:
            k = feat_map.shape[-1]
            size = feat_map.shape[1]
            image_belt = np.zeros((size, k * size))
            for j in range(k):
                feature_image = feat_map[0, :, :, j]
                # feature_image -= feature_image.mean()
                # feature_image /= feature_image.std()
                # feature_image *= 64
                # feature_image += 128
                # feature_image = np.clip(x, 0, 255).astype('uint8')
                image_belt[:, j * size: (j + 1) * size] = feature_image
            scale = total_width / k
            fig = plt.figure(figsize=(scale * k, scale))
            ax = fig.add_subplot()
            ax.title.set_text(layer_name + f' ({k} images)')
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85)
            ax.axis('off')
            ax.grid(False)
            ax.imshow(image_belt, aspect='auto')
            fig.savefig(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}/{layer_name}.jpg')
            fig.clf()
    fig1.savefig(f'{model_dir}/Features/{basename(model_name).removesuffix(".h5")}/prediction.jpg')
    plt.close('all')


def func():
    pass


def main():
    feature_map(model_dir='models/Beta', data_dir='new data')


if __name__ == '__main__':
    main()
