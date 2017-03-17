import pytest

from preprocessing.image_loader import BRATSDataLoader
from preprocessing.volume_image import VolumeImageDataGenerator

import yaml
with open("tests/args.yml", 'r') as stream:
    try:
        config_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


@pytest.fixture
def b_loader():
    return BRATSDataLoader(**config_args['brats_data_loader']['train'])


@pytest.fixture
def b_loader_val():
    return BRATSDataLoader(**config_args['brats_data_loader']['val'])


def test_b_loader(b_loader):
    patients, classes = b_loader.get_patients_classes()
    print zip(patients[:20], classes[:20])
    assert len(patients) == len(classes)
    image = b_loader.load(patients[0])
    assert image.shape == (240, 240, 155, 1)
    label = b_loader.load_label(patients[0])
    assert label.shape == (240, 240, 155, 1)


@pytest.fixture
def vol_datagen():
    return VolumeImageDataGenerator(
            **config_args['volume_image_data_generator']['train'])


def test_vol_datagen(vol_datagen, b_loader, b_loader_val):
    print('Train')
    train_iter_args = config_args[
            'volume_image_data_generator']['flow_from_loader']
    train_iter_args['volume_image_data_loader'] = b_loader
    train_generator = vol_datagen.flow_from_loader(**train_iter_args)
    for i in range(16):
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (1, 240, 240, 155, 1)
    print('Test')
    test_iter_args = config_args[
            'volume_image_data_generator']['flow_from_loader']
    test_iter_args['volume_image_data_loader'] = b_loader_val
    test_generator = vol_datagen.flow_from_loader(**test_iter_args)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (1, 240, 240, 155, 1)
