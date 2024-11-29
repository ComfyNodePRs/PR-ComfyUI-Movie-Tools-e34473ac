# Movie Tools for ComfyUI [Work in progress]

**Movie Tools** is a set of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed to simplify saving and loading batches of images with enhanced functionality like subfolder management, metadata embedding, and batch image handling.

![Screenshot](screen.png)


## Features

- **Save Images to Subfolders**  
  Save generated images to structured subfolders with customizable filenames and metadata embedding.

- **Load Images from Subdirectories**  
  Load images in batches from nested directories, with options to limit the number of images and specify a starting index.

---

## Node Descriptions

### 1. **SaveImagesWithSubfolder**
This node saves input images to a specified output directory, organizing them into subfolders and embedding metadata such as prompts and extra information.

#### **Input Parameters**
- `images`: Images to save.  
- `filename_prefix`: Prefix for saved files. Can include formatting like `%date:yyyy-MM-dd%`.  
- `subfolder_name`: Name of the subfolder to create within the output directory.  
- `prompt`: (Hidden) Prompt text for metadata embedding.  
- `extra_pnginfo`: (Hidden) Additional metadata to include in the saved PNG files.

#### **Output**
- Saves images to the specified location and returns metadata about the saved images.

---

### 2. **LoadImagesFromSubdirsBatch**
This node loads images from a directory and its subdirectories into a batch tensor, ensuring the images have consistent dimensions.

#### **Input Parameters**
- `directory`: Path to the directory containing images.  
- `image_load_cap`: (Optional) Limit on the number of images to load.  
- `start_index`: (Optional) Index from which to start loading images.

#### **Output**
- Returns a batch of loaded images as a tensor.

---

## Installation

This Node is designed for use within ComfyUI. Ensure ComfyUI is installed and operational in your environment. 

 - Copy the `ComfyUI-Movie-Tools` folder into the `custom_nodes` directory accessible by your ComfyUI project.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-tools-comfyui.git
   ```
2. Navigate to the directory:
   ```bash
   cd movie-tools-comfyui
   ```
3. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.8+
- Torch
- NumPy
- Pillow (PIL)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.
