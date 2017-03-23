import yaml
import pytest
from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import BRATSDataLoader


@pytest.fixture
def init_args():
    with open("tests/args.yml", 'r') as stream:
        try:
            init_args = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return init_args


@pytest.fixture
def vol_datagen(init_args):
    return VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['train'])


@pytest.fixture
def vol_loader_train(init_args):
    return BRATSDataLoader(
        **init_args['brats_data_loader']['train'])


@pytest.fixture
def vol_loader_val(init_args):
    return BRATSDataLoader(
        **init_args['brats_data_loader']['val'])


def test_init(vol_datagen):
    vol_datagen.image_shape == (240, 240, 155, 1)


def test_flow_from_loader_train(init_args, vol_datagen, vol_loader_train):
    train_iter_args = init_args['volume_image_data_generator'][
         'flow_from_loader']
    train_iter_args['volume_image_data_loader'] = vol_loader_train
    train_generator = vol_datagen.flow_from_loader(**train_iter_args)
    for i in range(16):
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (1, 240, 240, 155, 1)
        assert batch_y.shape == (1, 240, 240, 155, 5)


def test_flow_from_loader_val(init_args, vol_datagen, vol_loader_val):
    test_iter_args = init_args['volume_image_data_generator'][
        'flow_from_loader']
    test_iter_args['volume_image_data_loader'] = vol_loader_val
    test_generator = vol_datagen.flow_from_loader(**test_iter_args)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (1, 240, 240, 155, 1)
        assert batch_y.shape == (1, 240, 240, 155, 5)
