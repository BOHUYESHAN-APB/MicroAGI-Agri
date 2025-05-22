import unittest
import numpy as np
from microai_core.utils.data_augmentation import MicroAugment

class TestMicroAugment(unittest.TestCase):
    def setUp(self):
        self.augmentor = MicroAugment()
        self.test_img = np.random.rand(256, 256, 3) * 255

    def test_augmentation_output_shape(self):
        augmented = self.augmentor(self.test_img)
        self.assertEqual(augmented.shape, self.test_img.shape)

    def test_augmentation_values(self):
        augmented = self.augmentor(self.test_img)
        self.assertFalse(np.array_equal(augmented, self.test_img))

if __name__ == '__main__':
    unittest.main()