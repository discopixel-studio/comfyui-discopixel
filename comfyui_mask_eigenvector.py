from datetime import datetime

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator

@register_node("MaskToEigenvector", "Mask to Eigenvector")
class MaskToEigenvector:
    """
    Takes a mask as input, and calculates the centroid.
    Useful to find the center of a shape in the mask.
    This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.
    The eigenvector is calculated assuming that an identity oval would be vertically oriented.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Geometry"
    FUNCTION = "calculate_eigenvector"

    def calculate_eigenvector(self, image):
        image = 1.0 - image
        return (image,)