import cv2
import numpy as np
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator

@register_node("TransformImageToMatchMask", "Transform Image to Match Mask")
class TransformImageToMatchMask:
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
                "mask": ("MASK",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Geometry"
    FUNCTION = "transform_image"

    def convert_image_from_tensor_to_numpy(self, image_tensor):
        # Convert from PyTorch tensor to NumPy array
        image_np = image_tensor.squeeze().numpy()

        # Convert from normalized floats back to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Convert from RGB to BGR (OpenCV uses BGR)
        image_np = image_np[:, :, [2, 1, 0]]

        return image_np

    def convert_image_from_numpy_to_tensor(self, image_np):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Normalize the pixel values to [0.0, 1.0]
        image_normalized = image_rgb.astype(np.float32) / 255.0

        # Convert to a PyTorch tensor
        image_tensor = torch.from_numpy(image_normalized)

        # Add a batch dimension with [None,] or .unsqueeze(0)
        image_tensor = image_tensor[None,]

        return image_tensor

    def calculate_transformation(self, mask, image):
        # Convert the tensor to a numpy array and remove the batch and color dimensions
        # The resulting array will have shape [height, width]
        np_image = mask.squeeze().numpy()

        # Assuming the object is white and the background is black
        # Create a binary image (you might need to adjust the thresholding logic based on your image)
        binary_image = np_image > 0.5  # Simple thresholding for demonstration

        # Find the indices of non-zero (white) pixels
        y_indices, x_indices = np.nonzero(binary_image)

        # Calculate the centroid
        centroid_x, centroid_y = int(np.mean(x_indices)), int(np.mean(y_indices))

        # Calculate the covariance matrix
        cov_matrix = np.cov(x_indices, y_indices)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The largest eigenvalue corresponds to the major axis
        major_axis_length = 2 * np.sqrt(eigenvalues[1]) * 2  # Scale as needed
        minor_axis_length = 2 * np.sqrt(eigenvalues[0]) * 2  # Scale as needed

        # Angle between x-axis and the major axis of the ellipse in degrees
        rotation = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

        # Draw the ellipse
        ellipse_length = int(major_axis_length / 2)
        ellipse_width = int(minor_axis_length / 2)

        # Starting and ending dimensions
        height_out, width_out = mask.shape[1:3]
        height_in, width_in = image.shape[1:3]

        # Calculate the scale relative to the original template size
        scale_x = ellipse_width / 72
        scale_y = ellipse_length / 100

        # Center of the image
        cx, cy = int(0.5 * width_out), int(0.5 * height_out)

        # Translation from center of the image
        translate_x = centroid_x - cx
        translate_y = centroid_y - cy
        print(f"Centroid: {(centroid_x, centroid_y)}")
        print(f"Translate: {(translate_x, translate_y)}")


        # Initial Translation matrix (to origin)
        initial_translation_matrix = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        # Scaling matrix
        scaling_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])

        # Rotation matrix
        adjusted_rotation = rotation - np.pi / 2 # For whatever reason, the original is off by 90deg
        rotation_matrix = np.array([
            [np.cos(adjusted_rotation), -np.sin(adjusted_rotation), 0],
            [np.sin(adjusted_rotation), np.cos(adjusted_rotation), 0],
            [0, 0, 1]
        ])

        # Final Translation matrix (to new location)
        final_translation_matrix = np.array([
            [1, 0, centroid_x],
            [0, 1, centroid_y],
            [0, 0, 1]
        ])

        # Combine the matrices
        matrix = np.dot(final_translation_matrix, np.dot(rotation_matrix, np.dot(scaling_matrix, initial_translation_matrix)))

        print(f"Centroid: {(centroid_x, centroid_y)}")
        print(f"Ellipse: {(ellipse_length, ellipse_width)}")
        print(f"Rotation: {rotation}")
        print(f"Matrix: \n{matrix}")

        return (
            centroid_x,
            centroid_y,
            ellipse_length,
            ellipse_width,
            rotation,
            matrix,
        )

    def transform_image(self, mask, image):
        centroid_x, centroid_y, ellipse_length, ellipse_width, rotation, matrix = self.calculate_transformation(mask, image)

        # Create a blank canvas the same size as the mask
        height_in, width_in = image.shape[1:3]
        height_out, width_out = mask.shape[1:3]
        print(f"image.shape(): {(image.shape)} mask.shape(): {(mask.shape)}")
        print(f"image.shape(): {(height_in, width_in)} mask.shape(): {(height_out, width_out)}")
        canvas = np.zeros((height_out, width_out, 3), np.uint8)
        print(f"canvas.shape(): {canvas.shape} image.shape(): {image.shape}")

        # Drop the image into the center of this canvas
        x_offset = (width_out - width_in) // 2
        y_offset = (height_out - height_in) // 2
        image_np = self.convert_image_from_tensor_to_numpy(image)
        canvas[y_offset:y_offset + height_in, x_offset:x_offset + width_in] = image_np


        print(f"Matrix = \n{matrix}")



        # Apply the affine transformation
        transformed_image = cv2.warpAffine(canvas, matrix[:2], (width_out, height_out))

        centroid = (centroid_x, centroid_y)
        axes = (ellipse_length, ellipse_width)
        angle = rotation * (180 / np.pi)
        cv2.ellipse(transformed_image, centroid, axes, angle, 0, 360, (0, 255, 0), 2)        


        # print(f"Centroid = {(centroid_x, centroid_y)}")
        # cv2.circle(preview_image, (centroid_x, centroid_y), radius=5, color=(0, 0, 255), thickness=-5)
        # transformed_image = self.convert_image_from_numpy_to_tensor(transformed_image)
        final_image = self.convert_image_from_numpy_to_tensor(transformed_image)
        return (final_image,)
