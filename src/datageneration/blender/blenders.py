"""
Module with none cropper

Classes:
    NoneCropper
"""
from .alpha_blender import AlphaBlender
from .no_blend import NoBlender
from .pyramid_blender import PyramidBlender
from .seamless_blender import SeamlessBlender

available_blenders = {
    "no-blend": NoBlender(),
    "alpha": AlphaBlender(),
    "seamless": SeamlessBlender(),
    "pyramid": PyramidBlender(),
}
