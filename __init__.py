
import numpy as np
import folder_paths
import json
import os
import torch
from comfy.cli_args import args
from comfy.utils import common_upscale
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps

class SaveImagesWithSubfolder:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "subfolder_name": ("STRING", {"default": "", "tooltip": "Create subfolder for this images."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def run(self, images, filename_prefix="ComfyUI", subfolder_name="", prompt=None, extra_pnginfo=None):
        filename_prefix = filename_prefix.replace("%subfolder_name%", str(subfolder_name))
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))       
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class LoadImagesFromSubdirsBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "run"

    CATEGORY = "image"

    def run(self, directory: str, image_load_cap: int = 0, start_index: int = 0):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        def collect_image_paths(dir_path):
            image_paths = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
            return image_paths

        dir_files = collect_image_paths(directory)

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []

        if image_load_cap > 0:
            dir_files = dir_files[:image_load_cap]

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)

        if len(images) == 1:
            return (images[0], )

        elif len(images) > 1:
            image1 = images[0]

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            return (image1, )
      
    
NODE_CLASS_MAPPINGS = {
    "SaveImagesWithSubfolder": SaveImagesWithSubfolder,
    "LoadImagesFromSubdirsBatch": LoadImagesFromSubdirsBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagesWithSubfolder": "[Movie Tools] Save images",
    "LoadImagesFromSubdirsBatch": "[Movie Tools] Load images from subdirs",
}