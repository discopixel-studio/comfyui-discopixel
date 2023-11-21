# ComfyUI Geometry Nodes

A small collection of custom nodes for use with [ComfyUI](https://github.com/comfyanonymous/ComfyUI), for geometry calculations

## 🔌 Available Nodes

#### 1. Mask to Centroid

Takes a mask as input, and calculates the centroid.
Useful to find the center of a shape in the mask.
This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.

#### 2. Mask to Eigenvector

Takes a mask as input, and calculates the 2D eigenvector.
Useful to find the rotation and scaling of a shape in the mask.
This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.
Eigenvector is calculated assuming that an "identity" shape is vertically oriented oval.

## ⚙️ Installation

#### Method 1: Download each file individually

Go though each file and see which nodes you want to use. Download the corresponding file and put it in:

```
ComfyUI/custom_nodes
```

#### Method 2: Clone the whole repo

Install all nodes into your repo in one fell swoop.

```
cd ComfyUI/custom_nodes
git clone https://github.com/jamesWalker55/comfyui-various
```
