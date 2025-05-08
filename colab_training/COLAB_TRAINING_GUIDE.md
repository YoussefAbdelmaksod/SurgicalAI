# SurgicalAI Training on Google Colab

This guide helps you set up and run SurgicalAI training on Google Colab.

## Error Resolution

If you're seeing errors like:
```
python3: can't open file '/content/training/train_tool_detection.py': [Errno 2] No such file or directory
```

This is because the training scripts are in the wrong location. Here's how to fix it:

## Quick Fix

1. Upload the `fix_colab_paths.py` script to your Colab environment
2. Run it with: `!python fix_colab_paths.py`
3. Change to the root directory: `%cd /content`
4. Run the training scripts directly from the `/content` directory

## Fixed Training Commands

After running the fix script, use these commands:

```python
# Tool Detection Training
!python training/train_tool_detection.py \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 4 \
  --num_epochs 10 \
  --learning_rate 3e-4 \
  --backbone resnet50 \
  --use_mixed_precision True

# Phase Recognition Training
!python training/train_phase_recognition.py \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 2 \
  --num_epochs 10 \
  --vit_model vit_base_patch16_224 \
  --freeze_vit True

# Mistake Detection Training
!python training/train_all_models.py \
  --train_subset mistake_detection \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 4 \
  --num_epochs 8
```

## Complete Workflow

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Create a weights directory:
   ```python
   !mkdir -p /content/drive/MyDrive/SurgicalAI_clone/weights
   ```

3. Clone the repository:
   ```python
   !git clone https://github.com/YOUR_USERNAME/SurgicalAI SurgicalAI_clone
   %cd SurgicalAI_clone
   ```

4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   !pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
   ```

5. Copy and run the fix script:
   ```python
   %%writefile fix_paths.py
   # Paste the content of fix_colab_paths.py here
   
   !python fix_paths.py
   ```

6. Change to root directory and run training:
   ```python
   %cd /content
   # Run training commands as above
   ```

7. Save models to Drive:
   ```python
   !cp models/weights/tool_detection/tool_detection.pth /content/drive/MyDrive/SurgicalAI_clone/weights/
   !cp models/weights/vit_lstm/phase_recognition.pth /content/drive/MyDrive/SurgicalAI_clone/weights/
   !cp models/weights/mistake_detector/mistake_detection.pth /content/drive/MyDrive/SurgicalAI_clone/weights/
   ```

## Troubleshooting

If you still encounter issues:

1. Check that all paths are correct after running `fix_colab_paths.py`
2. Make sure your data directories contain the required training data
3. Check for any missing dependencies or Python modules
4. Examine GPU memory usage if you get CUDA out-of-memory errors (reduce batch size if needed)

For further assistance, refer to the project documentation or open an issue on GitHub. 