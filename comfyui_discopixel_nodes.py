import cv2
import numpy as np
import torch
import comfy
from segment_anything import SamPredictor
from skimage.draw import disk

import requests
from PIL import Image, ImageOps, ImageSequence
from io import BytesIO
import time

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator


def convert_image_from_numpy_to_tensor(image_np):
    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values to [0.0, 1.0]
    image_normalized = image_rgb.astype(np.float32) / 255.0

    # Convert to a PyTorch tensor
    image_tensor = torch.from_numpy(image_normalized)

    # Add a batch dimension with [None,] or .unsqueeze(0)
    image_tensor = image_tensor[None,]

    return image_tensor


# @register_node("OpenPoseToClothesMask", "OpenPose to Clothes Mask")
class OpenPoseToClothesMask:
    """
    Takes an OpenPose preprocessor POSE_KEYPOINT output and uses it to segment
    the clothes from the corresponding input image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "pose_keypoints": ("POSE_KEYPOINT",),
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    CATEGORY = "Discopixel"
    FUNCTION = "run"

    def combine_masks2(self, masks):
        if len(masks) == 0:
            return None
        else:
            initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
            combined_cv2_mask = initial_cv2_mask

            for i in range(1, len(masks)):
                cv2_mask = np.array(masks[i]).astype(np.uint8)

                if combined_cv2_mask.shape == cv2_mask.shape:
                    combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
                else:
                    # do nothing - incompatible mask
                    pass

            mask = torch.from_numpy(combined_cv2_mask)
            return mask

    def dilate_mask(self, mask, dilation_factor, iter=1):
        if dilation_factor == 0:
            return mask

        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

        if dilation_factor > 0:
            result = cv2.dilate(mask, kernel, iter)
        else:
            result = cv2.erode(mask, kernel, iter)

        return result

    def sam_predict(self, predictor, points, plabs, threshold):
        point_coords = None if not points else np.array(points)
        point_labels = None if not plabs else np.array(plabs)

        print(f"(custom sam_predict) points: {points}")
        print(f"(custom sam_predict) plabs: {plabs}")

        cur_masks, scores, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels
        )

        total_masks = []

        selected = False
        max_score = 0
        for idx in range(len(scores)):
            if scores[idx] > max_score:
                max_score = scores[idx]
                max_mask = cur_masks[idx]

            if scores[idx] >= threshold:
                selected = True
                total_masks.append(cur_masks[idx])
            else:
                pass

        if not selected:
            total_masks.append(max_mask)

        return total_masks

    # BODY KEYPOINTS
    # keypoint_head = 0
    # keypoint_neck = 1
    # keypoint_right_shoulder = 2
    # keypoint_right_elbow = 3
    # keypoint_right_wrist = 4
    # keypoint_left_shoulder = 5
    # keypoint_left_elbow = 6
    # keypoint_left_wrist = 7
    # keypoint_right_hip = 8
    # keypoint_right_knee = 9
    # keypoint_right_ankle = 10
    # keypoint_left_hip = 11
    # keypoint_left_knee = 12
    # keypoint_left_ankle = 13

    # FACE KEYPOINTS
    # keypoint_left_eye = 69
    # keypoint_right_eye = 68
    # keypoint_left_lip = 54
    # keypoint_right_lip = 48

    # HAND KEYPOINTS
    # keypoint_wrist = 0
    # keypoint_middle_knuckle = 9
    def point_from_pose(
        self, pose_keypoints, part, joint_key_a=None, joint_key_b=None, interpolate=0.5
    ):
        part_keypoints = pose_keypoints[0]["people"][0][part]

        if joint_key_a is None and joint_key_b is None:
            raise ValueError("Both joint_a and joint_b are None")

        a = (
            None
            if joint_key_a is None
            else part_keypoints[joint_key_a * 3 : joint_key_a * 3 + 2]
        )
        b = (
            None
            if joint_key_b is None
            else part_keypoints[joint_key_b * 3 : joint_key_b * 3 + 2]
        )

        if a is None and b is not None:
            return (b[0], b[1])
        elif b is None and a is not None:
            return (a[0], a[1])
        else:
            midpoint = (
                (a[0] * (1 - interpolate) + b[0] * interpolate),
                (a[1] * (1 - interpolate) + b[1] * interpolate),
            )
            return midpoint

    def preview_points(self, image_np, points_float, plabs):
        image_tensor_batch = convert_image_from_numpy_to_tensor(image_np)
        image_tensor = image_tensor_batch[0]

        # Check image shape (H, W, C) with 3 channels for RGB
        if image_tensor.dim() != 3 or image_tensor.shape[2] != 3:
            raise ValueError(
                f"Image must have shape (H, W, 3), got {image_tensor.shape}"
            )

        # Define colors
        colors = {0: [255, 0, 0], 1: [0, 0, 255]}  # Red for 0, Blue for 1

        for (x, y), label in zip(points_float, plabs):
            # Convert normalized coordinates to pixel coordinates
            height, width = image_tensor.shape[:2]
            px = int(x * width)
            py = int(y * height)
            color = torch.tensor(colors[label], dtype=image_tensor.dtype)

            # Draw circle
            rr, cc = disk((py, px), 16, shape=image_tensor.shape[:-1])
            image_tensor[rr, cc] = color

        return image_tensor[None,]  # Add batch dimension

    def run(self, sam_model, pose_keypoints, image, threshold):

        if sam_model.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            sam_model.to(device=device)

        try:
            predictor = SamPredictor(sam_model)
            image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
            predictor.set_image(image, "RGB")

            # _face = "face_keypoints_2d"
            _l_hand = "hand_left_keypoints_2d"
            _r_hand = "hand_right_keypoints_2d"
            _body = "pose_keypoints_2d"
            head = self.point_from_pose(pose_keypoints, _body, 0, 1)
            l_hand = self.point_from_pose(pose_keypoints, _l_hand, 0, 9)
            r_hand = self.point_from_pose(pose_keypoints, _r_hand, 0, 9)
            l_foot = self.point_from_pose(pose_keypoints, _body, 13, 12, -0.2)
            r_foot = self.point_from_pose(pose_keypoints, _body, 10, 9, -0.2)
            l_shin1 = self.point_from_pose(pose_keypoints, _body, 13, 12, 0.15)
            r_shin1 = self.point_from_pose(pose_keypoints, _body, 10, 9, 0.15)
            l_shin2 = self.point_from_pose(pose_keypoints, _body, 13, 12, 0.3)
            r_shin2 = self.point_from_pose(pose_keypoints, _body, 10, 9, 0.3)
            l_shin3 = self.point_from_pose(pose_keypoints, _body, 13, 12, 0.5)
            r_shin3 = self.point_from_pose(pose_keypoints, _body, 10, 9, 0.5)
            l_shin4 = self.point_from_pose(pose_keypoints, _body, 13, 12, 0.7)
            r_shin4 = self.point_from_pose(pose_keypoints, _body, 10, 9, 0.7)
            l_knee = self.point_from_pose(pose_keypoints, _body, 12, None)
            r_knee = self.point_from_pose(pose_keypoints, _body, 9, None)
            l_thigh = self.point_from_pose(pose_keypoints, _body, 12, 11, 0.5)
            r_thigh = self.point_from_pose(pose_keypoints, _body, 9, 8, 0.5)
            crotch = self.point_from_pose(pose_keypoints, _body, 11, 8)
            points_float = [
                head,
                l_hand,
                r_hand,
                l_foot,
                r_foot,
                l_shin1,
                r_shin1,
                l_shin2,
                r_shin2,
                l_shin3,
                r_shin3,
                l_shin4,
                r_shin4,
                l_knee,
                r_knee,
                l_thigh,
                r_thigh,
                crotch,
            ]
            h, w = image.shape[:2]
            points_pixels = [(p[0] * w, p[1] * h) for p in points_float]
            points = points_pixels
            # points = [tup for tup in points for _ in range(9)]
            plabs = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # plabs = [tup for tup in plabs for _ in range(9)]

            # points = [(0.01, 0.01), (0.5, 0.5)]
            # points = [tup for tup in points for _ in range(9)]
            # plabs = [0, 1]
            # plabs = [tup for tup in plabs for _ in range(9)]

            print(f"points: {points}")
            print(f"plabs: {plabs}")

            detected_masks = self.sam_predict(predictor, points, plabs, threshold)

            for i in range(len(detected_masks)):
                print(f"detected_mask {i}: {detected_masks[i].shape}")

            mask = self.combine_masks2(detected_masks)

        finally:
            if sam_model.is_auto_mode:
                print(f"semd to {device}")
                sam_model.to(device="cpu")

        if mask is not None:
            mask = mask.float()
            mask = self.dilate_mask(mask.cpu().numpy(), 0)
            mask = torch.from_numpy(mask)
        else:
            mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")  # empty mask

        preview = self.preview_points(image, points_float, plabs)

        return (mask, preview)


# @register_node("TransformTemplateOntoFaceMask", "Transform Template onto Face Mask")
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
    CATEGORY = "Discopixel"
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
        rotation += np.pi / 2  # Reverse the orientation compensation

        # Adjust the rotation to be between -90 and 90 degrees from the vertical (can't figure out why we need this)
        if rotation > np.pi * 0.75 and rotation < np.pi:
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
        initial_translation_matrix = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

        # Scaling matrix
        scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        # Rotation matrix
        # For whatever reason, the original is off by 90deg
        adjusted_rotation = rotation + np.pi / 2
        rotation_matrix = np.array(
            [
                [np.cos(adjusted_rotation), -np.sin(adjusted_rotation), 0],
                [np.sin(adjusted_rotation), np.cos(adjusted_rotation), 0],
                [0, 0, 1],
            ]
        )

        # Final Translation matrix (to new location)
        final_translation_matrix = np.array(
            [[1, 0, centroid_x], [0, 1, centroid_y], [0, 0, 1]]
        )

        # Combine the matrices
        matrix = np.dot(
            final_translation_matrix,
            np.dot(rotation_matrix, np.dot(scaling_matrix, initial_translation_matrix)),
        )

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
        # Ensure batch dimension for consistent handling
        if face_mask.dim() == 2:
            face_mask = face_mask.unsqueeze(0)
        if template_image.dim() == 3:
            template_image = template_image.unsqueeze(0)
        if template_mask.dim() == 2:
            template_mask = template_mask.unsqueeze(0)

        # Calculate the transformation
        centroid_x, centroid_y, ellipse_length, ellipse_width, rotation, matrix = (
            self.calculate_transformation(face_mask, template_mask)
        )

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
        image_canvas[
            y_offset : y_offset + height_in, x_offset : x_offset + width_in
        ] = template_image_np
        mask_canvas[y_offset : y_offset + height_in, x_offset : x_offset + width_in] = (
            template_mask_np
        )

        # Apply the affine transformation
        transformed_template_image = cv2.warpAffine(
            image_canvas, matrix[:2], (width_out, height_out)
        )
        transformed_template_mask = cv2.warpAffine(
            mask_canvas, matrix[:2], (width_out, height_out)
        )

        # Draw the ellipse to preview the calculation
        debug_calculation = False
        if debug_calculation:
            centroid = (centroid_x, centroid_y)
            axes = (ellipse_length, ellipse_width)
            angle = rotation * (180 / np.pi)
            cv2.ellipse(
                transformed_template_image,
                centroid,
                axes,
                angle,
                0,
                350,
                (0, 255, 0),
                2,
            )
            cv2.circle(
                transformed_template_image,
                (centroid_x, centroid_y),
                radius=5,
                color=(255, 0, 0),
                thickness=-5,
            )

        # Invert the mask back to normal
        transformed_template_mask = 255.0 - transformed_template_mask

        # Convert the image back to a tensor
        final_image = convert_image_from_numpy_to_tensor(transformed_template_image)
        final_mask = self.convert_mask_from_numpy_to_tensor(transformed_template_mask)

        return (final_image, final_mask)


@register_node("PhotoroomRemoveBG", "Remove Background with Photoroom")
class PhotoroomRemoveBG:
    """
    Removes background from an input image using Photoroom's API.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "Discopixel"
    FUNCTION = "run"

    def remove_background(self, image_buffer, api_key):
        try:
            # Set up the POST request to send the image
            files = {"image_file": ("upload.png", image_buffer, "image/png")}
            headers = {"x-api-key": api_key}

            post_response = requests.post(
                "https://sdk.photoroom.com/v1/segment",
                files=files,
                headers=headers,
                stream=True,
            )
            post_response.raise_for_status()

            # Save the output to a file
            output_path = f"/tmp/photoroom-result-{int(time.time())}.png"
            with open(output_path, "wb") as out_file:
                out_file.write(post_response.content)

            # Convert the resulting image into the same image format that was originally inputted to run()
            image_buffer = BytesIO(post_response.content)
            return image_buffer
        except Exception as e:
            print(f"API Call Error: {e}")
            return None

    def run(self, images, api_key):
        print(f"images: {images}")

        if images.dim() == 3:
            images = images.unsqueeze(0)

        output_images = []
        output_masks = []

        for batch_number, image_tensor in enumerate(images):
            image_np = 255.0 * image_tensor.cpu().numpy()
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)

            # Remove the background
            output_buffer = self.remove_background(image_buffer, api_key)

            print("0000")
            # Load the result image
            final_image = Image.open(output_buffer)

            print("AAAA")

            for i in ImageSequence.Iterator(final_image):
                i = ImageOps.exif_transpose(i)
                if i.mode == "I":
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if "A" in i.getbands():
                    mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
