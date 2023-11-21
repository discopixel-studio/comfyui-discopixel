# comfyui-geometry

A small collection of custom nodes for use with ComfyUI, for geometry calculations

## Available Nodes

### Mask to Centroid

Takes a mask as input, and calculates the centroid.
Useful to find the center of a shape in the mask.
This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.

### Mask to Eigenvector

Takes a mask as input, and calculates the 2D eigenvector.
Useful to find the rotation and scaling of a shape in the mask.
This assumes there is only one shape, and that the shape is comprised of white pixels over a black background.
Eigenvector is calculated assuming that an "identity" oval is vertically oriented.

## Installation

### Method 1 (Recommended): Download each file individually

Go though each file and see which nodes you want to use. Download the corresponding file and put it in:

```
ComfyUI/custom_nodes
```

### Method 2: Clone the repo

Install all nodes into your repo in one fell swoop.

```
cd ComfyUI/custom_nodes
git clone https://github.com/jamesWalker55/comfyui-various
```
