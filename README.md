# ComfyUI Discopixel Nodes

A small collection of custom nodes for use with [ComfyUI](https://github.com/comfyanonymous/ComfyUI), by Discopixel.

## 🔌 Available Nodes

#### 1. Transform Template onto Face Mask

Takes a template and a face mask as inputs, and transforms the template to fit over the face.
It does this via calculating the centroid and eigenvectors of the Face.
Useful for placing things onto a face.
This assumes there is only one shape in the face mask, and such that the shape is comprised of white pixels over a black background.

## ⚙️ Installation

First, clone the repo into your ComfyUI extensions directory:

```
cd ComfyUI/custom_nodes
git clone https://github.com/discopixel-studio/comfyui-discopixel
```

Then, install the dependencies using your preferred method:

```
pip3 install -r requirements.txt
```

Then restart ComfyUI, and off you go! 🚀
