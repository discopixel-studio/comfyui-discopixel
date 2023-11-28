"""
@author: Anson Kao
@title: ComfyUI Geometry
@nickname: ComfyUI Geometry
@description: A small collection of custom nodes for use with ComfyUI, for geometry calculations
"""

import importlib

print(f"Loading ComfyUI Geometry!")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = [
    ".comfyui_geometry_nodes",
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