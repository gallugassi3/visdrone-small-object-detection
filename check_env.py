import torch
import cv2
import ultralytics

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")
print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"OpenCV:         {cv2.__version__}")
print(f"Ultralytics:    {ultralytics.__version__}")

# בדיקה אמיתית: חישוב על ה-GPU, לא רק זיהוי שלו
x = torch.rand(1000, 1000).cuda()
y = x @ x
print(f"GPU compute:    OK, result on {y.device}")