"""
@author: Anson Kao
@title: ComfyUI Discopixel
@nickname: ComfyUI Discopixel
@description: A small collection of custom nodes for use with ComfyUI, by Discopixel
"""

import importlib

print(f"Loading ComfyUI Discopixel nodes!")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = [
    ".comfyui_discopixel_nodes",
]

def load_nodes(module_name: str):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    module = importlib.import_module(module_name, package=__name__)

    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        **module.NODE_CLASS_MAPPINGS,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **module.NODE_DISPLAY_NAME_MAPPINGS,
    }

for module_name in NODE_MODULES:
    load_nodes(module_name)