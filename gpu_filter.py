import cv2
import time
import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve2d  # GPU convolution

# Load grayscale image
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("sample.jpg not found!")

# Define a simple 3x3 blur kernel
kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]], dtype=np.float32) / 9.0

# ---------- CPU ----------
start = time.time()
out_cpu = cv2.filter2D(img, -1, kernel)
cpu_time = time.time() - start
print(f"CPU Time: {cpu_time*1000:.2f} ms")

# ---------- GPU ----------
img_gpu = cp.asarray(img, dtype=cp.float32)
kernel_gpu = cp.asarray(kernel, dtype=cp.float32)

start = time.time()
out_gpu = convolve2d(img_gpu, kernel_gpu, mode='same', boundary='symm')
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"GPU Time: {gpu_time*1000:.2f} ms")

# Convert GPU result to uint8 for saving
out_gpu = cp.asnumpy(out_gpu).astype(np.uint8)

# Save results
cv2.imwrite("output_cpu.jpg", out_cpu)
cv2.imwrite("output_gpu.jpg", out_gpu)


