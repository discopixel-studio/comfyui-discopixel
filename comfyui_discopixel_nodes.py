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

@register_node("TransformTemplateOntoFaceMask", "Transform Template onto Face Mask")
class TransformTemplateOntoFaceMask:
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
                "face_mask": ("MASK",),
                "template_image": ("IMAGE",),
                "template_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "Geometry"
    FUNCTION = "transform_template"

    def convert_image_from_tensor_to_numpy(self, image_tensor):
        # Convert from PyTorch tensor to NumPy array
        image_np = image_tensor.squeeze().numpy()

        # Convert from normalized floats back to uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Convert from RGB to BGR (OpenCV uses BGR)
        image_np = image_np[:, :, [2, 1, 0]]

        return image_np

    def convert_mask_from_tensor_to_numpy(self, mask_tensor):
        # Convert from PyTorch tensor to NumPy array
        mask_np = mask_tensor.squeeze().numpy()

        # Convert from normalized floats back to uint8
        mask_np = (mask_np * 255).astype(np.uint8)

        return mask_np

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

    def convert_mask_from_numpy_to_tensor(self, mask_np):
        # Normalize the pixel values to [0.0, 1.0]
        mask_normalized = mask_np.astype(np.float32) / 255.0

        # Convert to a PyTorch tensor
        mask_tensor = torch.from_numpy(mask_normalized)

        # Add a batch dimension with [None,] or .unsqueeze(0)
        mask_tensor = mask_tensor[None,]

        return mask_tensor

    def calculate_transformation(self, face_mask, template_mask):
        # Convert the tensor to a numpy array and remove the batch and color dimensions
        # The resulting array will have shape [height, width]
        np_image = self.convert_mask_from_tensor_to_numpy(face_mask)

        # Assuming the object is white and the background is black
        # Create a binary image (you might need to adjust the thresholding logic based on your image)
        binary_image = np_image > 0.5  # Simple thresholding for demonstration

        # Find the indices of non-zero (white) pixels
        y_indices, x_indices = np.nonzero(binary_image)

        # Calculate the centroid
        centroid_x, centroid_y = int(np.mean(x_indices)), int(np.mean(y_indices))

        # Re-orient the image so that the "top" is facing right, because linear algebra treats positive X-axis as 0 degrees, and we want rotation to be relative to the "top"
        oriented_image = np.rot90(binary_image, k=3)

        # Find the indices of non-zero (white) pixels
        y_indices, x_indices = np.nonzero(oriented_image)

        # Calculate the covariance matrix
        cov_matrix = np.cov(x_indices, y_indices)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The largest eigenvalue corresponds to the major axis
        major_axis_length = 2 * np.sqrt(eigenvalues[1]) * 2  # Scale as needed
        minor_axis_length = 2 * np.sqrt(eigenvalues[0]) * 2  # Scale as needed

        # Angle between x-axis and the major axis of the ellipse in degrees
        rotation = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        rotation += np.pi / 2 # Reverse the orientation compensation

        # Adjust the rotation to be between -90 and 90 degrees from the vertical (can't figure out why we need this)
        if (rotation > np.pi * 0.75 and rotation < np.pi):
            rotation -= np.pi

        # Draw the ellipse
        ellipse_length = int(major_axis_length / 2)
        ellipse_width = int(minor_axis_length / 2)

        # Starting and ending dimensions
        height_out, width_out = face_mask.shape[1:3]
        height_in, width_in = template_mask.shape[1:3]

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
        adjusted_rotation = rotation + np.pi / 2 # For whatever reason, the original is off by 90deg
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

        print("\n=== Transformation ===")
        print(f"Centroid: {(centroid_x, centroid_y)}")
        print(f"Ellipse: {(ellipse_length, ellipse_width)}")
        print(f"Rotation: {rotation * (180 / np.pi)}")
        print(f"Matrix: \n{matrix}")

        return (
            centroid_x,
            centroid_y,
            ellipse_length,
            ellipse_width,
            rotation,
            matrix,
        )

    def transform_template(self, face_mask, template_image, template_mask):
        centroid_x, centroid_y, ellipse_length, ellipse_width, rotation, matrix = self.calculate_transformation(face_mask, template_mask)

        # Create a blank canvas the same size as the mask
        height_in, width_in = template_image.shape[1:3]
        height_out, width_out = face_mask.shape[1:3]
        image_canvas = np.zeros((height_out, width_out, 3), np.uint8)
        mask_canvas = np.zeros((height_out, width_out), np.uint8)

        # Drop the image into the center of this canvas
        x_offset = (width_out - width_in) // 2
        y_offset = (height_out - height_in) // 2
        template_image_np = self.convert_image_from_tensor_to_numpy(template_image)
        template_mask_np = self.convert_mask_from_tensor_to_numpy(template_mask)

        # Invert the mask before transforming so unmasked space around the canvas is maintained automatically
        template_mask_np = 255.0 - template_mask_np

        # Drop the input template image and mask into the center of the output canvases
        image_canvas[y_offset:y_offset + height_in, x_offset:x_offset + width_in] = template_image_np
        mask_canvas[y_offset:y_offset + height_in, x_offset:x_offset + width_in] = template_mask_np

        # Apply the affine transformation
        transformed_template_image = cv2.warpAffine(image_canvas, matrix[:2], (width_out, height_out))
        transformed_template_mask = cv2.warpAffine(mask_canvas, matrix[:2], (width_out, height_out))

        # Draw the ellipse to preview the calculation
        debug_calculation = False
        if (debug_calculation):
            centroid = (centroid_x, centroid_y)
            axes = (ellipse_length, ellipse_width)
            angle = rotation * (180 / np.pi)
            cv2.ellipse(transformed_template_image, centroid, axes, angle, 0, 350, (0, 255, 0), 2)        
            cv2.circle(transformed_template_image, (centroid_x, centroid_y), radius=5, color=(255, 0, 0), thickness=-5)

        # Invert the mask back to normal
        transformed_template_mask = 255.0 - transformed_template_mask

        # Convert the image back to a tensor
        final_image = self.convert_image_from_numpy_to_tensor(transformed_template_image)
        final_mask = self.convert_mask_from_numpy_to_tensor(transformed_template_mask)

        return (final_image, final_mask)
