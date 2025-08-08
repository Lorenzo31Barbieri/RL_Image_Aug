# RL Image Evaluation Tools

Tools for running RL agent evaluation on CIFAR-10 images and saving both original and augmented versions.

## Files

- `config.py` - Configuration parameters (number of images, paths, etc.)
- `run_evaluation.py` - Main evaluation script
- `__init__.py` - Package initialization
- `README.md` - This documentation

## Usage

### From project root:
```bash
python -m rl_image_tools.run_evaluation
```

### From within the folder:
```bash
cd rl_image_tools
python run_evaluation.py
```

## Configuration

Edit `config.py` to change:

- `NUM_IMAGES` - Number of images to process (default: 100)
- `MAX_STEPS_PER_EPISODE` - RL agent steps per image (default: 3)
- `DEVICE` - 'cuda' or 'cpu'
- Model paths and output directories

## Output

The script creates:
- `./rl_image_tools/output_images/original/` - Original CIFAR-10 images
- `./rl_image_tools/output_images/augmented/` - RL-augmented images

## File Naming

Images are saved with descriptive names:
- `001_cat_improved_brightness_flip_1234.png`
  - `001` - Sequential number
  - `cat` - CIFAR-10 class name
  - `improved` - Status (improved/degraded/nochange)
  - `brightness_flip` - Actions taken by RL agent
  - `1234` - Original dataset index

## Requirements

- Trained VGG19 classifier at `./checkpoint/ckpt.pth`
- Optional: Trained RL agent at `./models/best_improved_dqn_model.pth`
- CIFAR-10 dataset (downloaded automatically)
