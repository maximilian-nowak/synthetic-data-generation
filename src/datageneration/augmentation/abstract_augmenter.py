"""
Module with abstract augmenter class

Classes:
    AbstractAugmenter
"""

from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractAugmenter:
    __metaclass__ = ABCMeta

    @abstractmethod
    def augment(self, image: np.ndarray, mask: np.ndarray):
        pass
