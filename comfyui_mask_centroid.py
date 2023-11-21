import cv2

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator

@register_node("MaskToCentroid", "Mask to Centroid")
class MaskToCentroid:
    """
    Takes a mask as input, and calculates the centroid.
    Useful to find the center of a shape in the mask.
    This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.
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
    FUNCTION = "calculate_centroid"

    def calculate_centroid(self, image):
        # Load the image in grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get a binary image
        _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

        # Convert to numpy array for vectorized operations
        np_image = np.array(binary_image)

        # Find indices of all non-zero elements (white pixels)
        y_indices, x_indices = np.nonzero(np_image)

        # Calculate the weighted average of the positions
        total_weight = len(x_indices)
        if total_weight == 0:
            return None

        centroid_x = np.sum(x_indices) // total_weight
        centroid_y = np.sum(y_indices) // total_weight

        preview_image = image.copy()
        cv2.circle(preview_image, (centroid_x, centroid_y), radius=5, color=(0, 0, 255), thickness=-5)

        # return {
        #     'centroid': (centroid_x, centroid_y)
        # }

        return (preview_image,)
