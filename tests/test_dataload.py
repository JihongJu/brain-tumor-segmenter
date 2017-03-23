import yaml
from nose.tools import (
    assert_equal,
    assert_not_equal,
    assert_raises)
from preprocessing.image_loader import BRATSDataLoader
from preprocessing.volume_image import VolumeImageDataGenerator


class TestVolumeImageDataGenerator():
    """Test VolumeImageDataGenerator class."""
    @classmethod
    def setup_class(klass):
        with open("tests/args.yml", 'r') as stream:
            try:
                klass._args = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @classmethod
    def teardown_class(klass):
        pass

    def setUp(self):
        pass

    def teardown(self):
        pass

    def test_init(self):
        pass

# @pytest.fixture
# def b_loader():
#     return BRATSDataLoader(**config_args['brats_data_loader']['train'])
#
#
# @pytest.fixture
# def b_loader_val():
#     return BRATSDataLoader(**config_args['brats_data_loader']['val'])
#
#
# def test_b_loader(b_loader):
#     patients, classes = b_loader.get_patients_classes()
#     print zip(patients[:20], classes[:20])
#     assert len(patients) == len(classes)
#     image = b_loader.load(patients[0])
#     assert image.shape == (240, 240, 155, 1)
#     label = b_loader.load_label(patients[0])
#     assert label.shape == (240, 240, 155, 1)
