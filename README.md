# DIV2K BPP-based Image Selection

This repository contains code for processing the DIV2K dataset to select diverse images based on Bits Per Pixel (BPP) values after applying super-resolution.

## Overview

The notebook processes the DIV2K dataset to:
1. Compute BPP (Bits Per Pixel) for all images after applying super-resolution
2. Sort images by BPP values
3. Select 10 diverse images based on BPP distribution
4. Apply full super-resolution processing only to selected images

## Requirements

- Python 3.8+
- PyTorch
- compressai
- transformers
- torchvision
- PIL/Pillow
- tqdm
- scikit-image

## Installation

```bash
pip install torch torchvision compressai transformers pillow tqdm scikit-image
```

## Usage

1. Run the notebook `process_div2k_bpp_selection.ipynb`
2. The notebook will automatically download the DIV2K dataset if not present
3. Process all images to compute BPP values
4. Select 10 diverse images
5. Results are saved to `div2k_bpp_sorted_selection.pkl`

## Models Used

- **Super-Resolution**: Swin2SR (from Hugging Face)
- **Compression**: cheng2020-attn (from CompressAI)

## Performance

- **CPU**: ~2+ hours for 800 images
- **GPU**: ~5-15 minutes for 800 images (with batching)

## Notebook Structure

- `process_div2k_bpp_selection.ipynb`: Main notebook for BPP-based image selection

## Notes

- The DIV2K dataset (~2GB) will be automatically downloaded if not present
- Processing is optimized with batching for better performance
- Results are saved in pickle format for further analysis
