# DeepSRS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/DeepRaman/ci.yml?branch=main)](https://github.com/yourusername/DeepRaman/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/downloads/)

A research project leveraging deep learning for **Stimulated Raman Scattering (SRS) super-resolution**. DeepRaman aims to push the boundaries of resolution in SRS microscopy, enabling more precise biological and chemical imaging than ever before.

---

## Table of Contents
- [DeepSRS](#deepsrs)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [References](#references)
  - [Contact](#contact)

---

## About the Project
- **Goal**: To develop and validate a deep neural network model that enhances SRS images beyond their native spatial resolution.  
- **Approach**: Combines advanced image reconstruction techniques, deep learning, and domain-specific knowledge in Raman spectroscopy.  
- **Applications**: Biological imaging, materials analysis, and other scientific fields that rely on Raman spectroscopy.

Here, you can include an overview of:
1. The main objective: super-resolve SRS images.  
2. The types of deep learning models or frameworks you explore (e.g., CNNs, GANs, Transformers).  
3. Why super-resolution is valuable for SRS imaging.

---

## Features
- **Super-Resolution Model**: Enhanced image details from low-resolution SRS acquisitions.  
- **Noise Reduction**: Integrated denoising methods to handle signal-to-noise issues inherent to Raman imaging.  
- **Flexible Architecture**: Pluggable modules to experiment with different deep learning backbones.  
- **Easy-to-Use API**: Straightforward interfaces for training, inference, and evaluation.

---

## Installation

1. **Clone the repository**:

~~~~bash
git clone https://github.com/RogueLiquid/DeepSRS.git
cd DeepRaman
~~~~

2. **Create a virtual environment (optional but recommended)**:

~~~~bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
~~~~

3. **Install dependencies**:

~~~~bash
pip install -r ESRGAN_requirements.txt
~~~~

~~~~bash
pip install -r ResShift_requirements.txt
~~~~

---

## Usage
Provide instructions for how to run the primary scripts or functions of DeepSRS. For example:

1. **Prepare your SRS data**:  
   - Organize your SRS data into a folder, the data format is tiff (ends with .tif).
   - Organize your processed data into a folder, the data format is png (ends with .png).

2. **Inference**:

   - To run our program for super-resolution, you can:

~~~~bash
# Run the autoencoder:
python run_autoencoder.py --input <input_tiff_folder> --output <output_tiff_folder>

# Run the ESRGAN:
python run_ESRGAN.py --input <input_image_folder> --output <output_image_folder>

# Run the ResShift
python run_ResShift.py --input <input_image_folder> --output <output_image_folder>
~~~~

---

## License
This project is licensed under the [MIT License](LICENSE).  
Feel free to add or modify license information as needed.

---

## References
If your work relies on or is inspired by certain papers or repositories, cite them here:  
1. Author et al., **Paper Title**, *Journal/Conference*, Year. [\[Link\]](https://example.com)  
2. Deep Learning Frameworks (e.g., PyTorch, TensorFlow).  
3. SRS imaging resources or datasets.

---

## Contact
Created by developers from ZJE – feel free to reach out!  
- **Email**: jiaxin1.22@intl.zju.edu.cn
- **GitHub**: [3220111903bit](https://github.com/3220111903bit)
- **Email**: songxiang.22@intl.zju.edu.cn
- **GitHub**: [Samuel Xu](https://github.com/RainbowBombs)
- **Email**: yihuan.22@intl.zju.edu.cn
- **GitHub**: [RogueLiquid](https://github.com/RogueLiquid)

If you find this project useful in your research, please consider [citing us](#references) or giving a star in GitHub.

---

*Thank you for checking out DeepRaman! We look forward to empowering the SRS imaging community with improved super-resolution techniques, and we hope this project sparks further innovation in biomedical optics and beyond.*
