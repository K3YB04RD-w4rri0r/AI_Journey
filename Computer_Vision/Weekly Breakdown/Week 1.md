# 🗓️ WEEK 1: PyTorch Fluency & Memory Mechanics

**Pace:** 2 Hours / Day | **Goal:** Build the environment, master tensor shape manipulations, understand how PyTorch allocates memory, and write hardware-agnostic code.

### ⬛ MONDAY: The Workspace & Hardware Linking (2 Hours)

_A pristine environment prevents dependency hell later when we deal with CUDA and TensorRT._

- [x] **1. Install Miniforge / Miniconda**
- [x] **2. Environment Creation & PyTorch Installation**
    - `conda create -n cv_env python=3.10 -y`
    - Go to [pytorch.org](http://pytorch.org/). Install the correct version for your hardware (CUDA for NVIDIA, MPS for Apple Silicon, CPU otherwise).
- [x] **3. Hardware Verification Script**
    - Write a short `check_device.py` script.
    - Assert that `torch.cuda.is_available()` or `torch.backends.mps.is_available()` is True.
    - Print the name of your GPU using `torch.cuda.get_device_name(0)` (if applicable).
- [x] **4. VS Code Tooling**
    - Install extensions: **Python**, **Jupyter**, and **Ruff** (linter).
    - Configure Ruff to format your code automatically on save.

### 🟥 TUESDAY: Tensor Instantiation & The Device Dance (2 Hours)

_Tensors are multi-dimensional arrays. Today is about making them, typing them, and moving them through the PCIe bus._

- [x] **1. Creation & Dtypes**
    - In a Jupyter notebook, create random tensors (`torch.randn`), zeros, and ones.
    - Experiment with `dtype`. Create a `torch.float32` tensor and a `torch.int64` tensor. Divide them. What happens to the type?
- [x] **2. Device Agnostic Code**
    - Write the standard PyTorch device string: `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
- [x] **3. CPU to GPU Benchmarking**
    - Create a massive `10000 x 10000` tensor on the CPU. Multiply it by itself. Time it using `time.time()`.
    - Move it to your device (`.to(device)`). Multiply it. Time it.
    - _(Note the massive speedup. You are now utilizing parallel GPU cores)._

### 🟧 WEDNESDAY: Under the Hood - Strides & Memory (2 Hours)

_This is the biggest hurdle for beginners. PyTorch tensors are just 1D blocks of memory masquerading as 3D/4D shapes. Today you learn how that works._

- [x] **1. View vs. Reshape**
    - Create `x = torch.arange(12)`.
    - Use `y = x.view(3, 4)` and `z = x.reshape(3, 4)`. They look the same.
- [x] **2. Contiguous Memory & Strides**
    - Transpose `y` using `y.t()`.
    - Try to do `.view(-1)` on the transposed tensor. **It will crash.**
    - _Why?_ Print `y.is_contiguous()` and `y.stride()`. Read up on why transposing a tensor changes how it is read from memory, breaking `.view()`.
    - Fix the crash using `.contiguous().view(-1)` or simply `.reshape(-1)`.

### 🟨 THURSDAY: The Shape Shifter - Permute, Squeeze, Mask (2 Hours)

_In Computer Vision, a single image is `[Channels, Height, Width]`. A batch is `[Batch, Channels, Height, Width]`. Matplotlib expects `[Height, Width, Channels]`. You will shift shapes constantly._

- [x] **1. Squeeze and Unsqueeze**
    - Create an image tensor `img = torch.randn(3, 256, 256)`.
    - Neural networks demand a batch dimension. Add one at index 0 using `.unsqueeze(0)`. (Shape: `[1, 3, 256, 256]`).
    - Remove it using `.squeeze(0)`.
- [x] **2. Permute**
    - Create a batch of 32 images: `batch = torch.randn(32, 3, 224, 224)`.
    - Rearrange the dimensions so the color channels are at the end (required for plotting). Use `.permute(0, 2, 3, 1)`.
- [x] **3. Masking & Slicing**
    - Slice the batch to extract only the top-left 100x100 pixels of all images.
    - Create a mask: set all values in the tensor `< 0` to `0` (this is manually coding a ReLU activation!).

### 🟩 FRIDAY: Advanced Math & Broadcasting (2 Hours)

_Broadcasting is the silent bug creator. It allows PyTorch to do math on tensors of different shapes, but if you don't understand it, it will corrupt your data without throwing an error._

- [x] **1. Broadcasting Mechanics**
    - Create an image batch: `batch = torch.randn(32, 3, 224, 224)`.
    - Create an RGB mean tensor: `means = torch.tensor([0.5, 0.5, 0.5])`.
    - Reshape `means` to `[1, 3, 1, 1]` so you can mathematically subtract it from `batch`.
- [x] **2. Reductions & Argmax**
    - Create a dummy output of a neural network: `preds = torch.randn(32, 10)` (32 images, 10 possible classes).
    - Get the predicted class for each image using `torch.argmax(preds, dim=1)`.
- [ ] **3. Introduction to Einsum (Senior Flex)**
    - Look up `torch.einsum`.
    - Write a matrix multiplication using einsum instead of `@`. (e.g., `torch.einsum('ik,kj->ij', A, B)`). This notation is heavily used in Vision Transformers (ViTs) later.

### 🟦 SATURDAY: Engineering - CLI & Scripts (2 Hours)

_Leave the Jupyter notebook. Real pipelines run from the terminal. Treat this as your first real portfolio commit._

- [ ] **1. Write `tensor_mechanics.py`**
    - Create a pure Python script.
    - Write a function `def simulate_forward_pass(batch_size: int):` that generates dummy images, flattens them (`.view(batch_size, -1)`), and multiplies them by a dummy weight matrix.
- [x] **2. Command Line Arguments**
    - Import `argparse`.
    - Make your script accept arguments from the terminal so you can run: `python tensor_mechanics.py --batch_size 64 --device cuda`.
- [x] **3. Clean & Lint**
    - Run the Ruff linter over your code: `ruff check tensor_mechanics.py`. Fix any warnings about unused imports or formatting.

### 🟪 SUNDAY: The Visual Proof & Git Sync (2 Hours)

_Bridge the gap between raw math and visual output._

- [x] **1. Tensor to Image**
    - In your script, generate a random noise tensor: `torch.rand(3, 256, 256)`.
    - Permute it to `[256, 256, 3]`.
    - Move it to the CPU, convert to NumPy (`.cpu().numpy()`).
    - Use `matplotlib.pyplot` to save it to disk as `noise.png`.
- [x] **2. Git Push**
    - Initialize Git (`git init`), create a `.gitignore` (ignore `__pycache__` and `.env`).
    - Push your repo to GitHub (`CV_32_Week_Journey`).
- [ ] **3. Review & Reflection**
    - Spend the last 45 minutes reviewing documentation. Type `help(torch.permute)` or `help(torch.view)` in your Python REPL. Get used to reading the official docs natively.