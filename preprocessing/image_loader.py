import os
import glob
import numpy as np
import SimpleITK as sitk
import keras.backend as K
from preprocessing.volume_image import (
        VolumeImageDataLoader)


class BRATSDataLoader(VolumeImageDataLoader):

    def get_patients_classes(self):
        patients = []
        classes = []
        for lvl in os.listdir(self.image_dir):
            for pat in os.listdir(os.path.join(self.image_dir, lvl)):
                mods = glob.glob(
                        os.path.join(self.image_dir, lvl, pat,
                                     '*MR*/*MR*.mha'))
                annot = glob.glob(
                        os.path.join(self.image_dir, lvl, pat,
                                     '*OT*/*OT*.mha'))
                patients += mods
                classes += annot * len(mods)
        return patients, classes

    def load(self, p):
        itk_img = sitk.ReadImage(p)
        img_array = sitk.GetArrayFromImage(itk_img)  # shape = (155, 240, 240)
        img_array = img_array.transpose(1, 2, 0)     # shape = (240, 240, 155)
        img_array = img_array.astype(K.floatx())
        if self.dim_ordering == 'tf':
            arr = img_array[..., np.newaxis]
        else:
            arr = img_array[np.newaxis, ...]
        return arr

    def load_label(self, p):
        arr = self.load(p)
        arr = arr.astype(np.int8)
        return arr
