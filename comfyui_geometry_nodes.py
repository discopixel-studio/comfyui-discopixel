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

    def calculate_transformation_matrix(self, mask, image):
        # Convert the tensor to a numpy array and remove the batch and color dimensions
        # The resulting array will have shape [height, width]
        np_image = mask.squeeze().numpy()

        # Assuming the object is white and the background is black
        # Create a binary image (you might need to adjust the thresholding logic based on your image)
        binary_image = np_image > 0.5  # Simple thresholding for demonstration

        # Find the indices of non-zero (white) pixels
        y_indices, x_indices = np.nonzero(binary_image)

        # Calculate the mean of the points
        x_mean, y_mean = np.mean(x_indices), np.mean(y_indices)

        # Calculate the covariance matrix
        cov_matrix = np.cov(x_indices, y_indices)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The largest eigenvalue corresponds to the major axis
        major_axis_length = 2 * np.sqrt(eigenvalues[1]) * 2  # Scale as needed
        minor_axis_length = 2 * np.sqrt(eigenvalues[0]) * 2  # Scale as needed

        # Angle between x-axis and the major axis of the ellipse in degrees
        angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]) * (180 / np.pi)

        # Draw the ellipse
        center = (int(x_mean), int(y_mean))
        axes = (int(major_axis_length / 2), int(minor_axis_length / 2))

        return (center, axes, angle)

        # # Adjust scale for the identity oval dimensions (using golden ratio)
        # scale_x = scale_x / 1.618033988749895
        # scale_y = scale_y / 1

        # Translation
        height, width = mask.shape[1:3]
        translate_x = centroid_x - 0.5 * width
        translate_y = centroid_y - 0.5 * height
        print(f"Centroid: {(centroid_x, centroid_y)}")
        print(f"Translate: {(translate_x, translate_y)}")

        # Center of the image
        cx, cy = width // 2, height // 2

        # Translate to origin matrix (3x3)
        translate_to_origin = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        # Translate back to center matrix (3x3)
        translate_back_to_center = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ])

        # 1. Scaling Matrix (3x3)
        FACTOR = 2 # Constant to adjust the scaling to match the edges of the mask
        scaling_matrix = np.array([
            [np.sqrt(eigenvalues[0]) * FACTOR / 72, 0, 0],
            [0, np.sqrt(eigenvalues[1]) * FACTOR / 100, 0],
            [0, 0, 1]
        ])
        scaling_matrix = np.dot(translate_back_to_center, np.dot(scaling_matrix, translate_to_origin))
        print(f"Scaling Matrix = \n{scaling_matrix}")

        # 2. Rotation Matrix (3x3)
        # angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle = rotation - np.pi / 2
        cos, sin = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        rotation_matrix = np.dot(translate_back_to_center, np.dot(rotation_matrix, translate_to_origin))
        print(f"Rotation Matrix = \n{rotation_matrix}")

        # 3. Combined Matrix (2x3)
        combined_matrix = np.dot(scaling_matrix, rotation_matrix)[:2]

        return combined_matrix

            # # Translate to origin
            # translate_to_origin = np.array([
            #     [1, 0, -width / 2],
            #     [0, 1, -height / 2],
            #     [0, 0, 1]
            # ])

            # # Scale the eigenvectors
            # scaled_eigenvector1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) / 144
            # scaled_eigenvector2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) / 200

            # # Form the combined rotation and scaling matrix
            # rotation_scaling_matrix = np.column_stack((scaled_eigenvector1, scaled_eigenvector2, [0, 0]))
            # rotation_scaling_matrix = np.vstack([rotation_scaling_matrix, [0, 0, 1]])

            # # Combine the transformations
            # combined_transform = np.dot(np.linalg.inv(translate_to_origin), rotation_scaling_matrix)
            # combined_transform = np.dot(combined_transform, translate_to_origin)

            # # The final matrix to use in cv2.warpAffine should exclude the last row
            # final_transform_matrix = combined_transform[:2, :]

            # return final_transform_matrix

        # # Scale the eigenvectors (incorporating scaling into rotation)
        # scaled_eigenvector1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) / 144
        # scaled_eigenvector2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) / 200

        # # Form the combined rotation and scaling matrix
        # rotation_scaling_matrix = np.column_stack((scaled_eigenvector1, scaled_eigenvector2))

        # # Create the final transformation matrix
        # transformation_matrix = np.identity(3)
        # # transformation_matrix[:2, :2] = rotation_scaling_matrix
        # transformation_matrix[:2, 2] = [translate_x, translate_y]

        # return transformation_matrix

    def transform_image(self, mask, image):
        center, axes, angle = self.calculate_transformation_matrix(mask, image)

        # Create a blank canvas the same size as the mask
        height_in, width_in = image.shape[1:3]
        height_out, width_out = mask.shape[1:3]
        print(f"image.shape(): {(image.shape)} mask.shape(): {(mask.shape)}")
        print(f"image.shape(): {(height_in, width_in)} mask.shape(): {(height_out, width_out)}")
        canvas = np.zeros((height_out, width_out, 3), np.uint8)
        print(f"canvas.shape(): {canvas.shape} image.shape(): {image.shape}")

        # # Drop the image into the center of this canvas
        # x_offset = (width_out - width_in) // 2
        # y_offset = (height_out - height_in) // 2
        # image_np = self.convert_image_from_tensor_to_numpy(image)
        # canvas[y_offset:y_offset + height_in, x_offset:x_offset + width_in] = image_np


        # print(f"Transform Matrix = \n{transformation_matrix}")



        # Apply the affine transformation
        # transformed_image = cv2.warpAffine(canvas, transformation_matrix[:2], (width_out, height_out))

        cv2.ellipse(canvas, center, axes, angle, 0, 360, (0, 255, 0), 2)        


        # print(f"Centroid = {(centroid_x, centroid_y)}")
        # cv2.circle(preview_image, (centroid_x, centroid_y), radius=5, color=(0, 0, 255), thickness=-5)
        # transformed_image = self.convert_image_from_numpy_to_tensor(transformed_image)
        transformed_image = self.convert_image_from_numpy_to_tensor(canvas)
        return (transformed_image,)
