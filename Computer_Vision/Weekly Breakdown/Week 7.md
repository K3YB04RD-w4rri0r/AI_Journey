# 🗓️ WEEK 7: Transfer Learning, Network Surgery & Extreme Augmentation
**Pace:** 2 Hours / Day | **Goal:** Master state dictionary manipulation, freeze computational graphs, implement differential learning rates, and integrate high-performance `albumentations` pipelines.

### ⬛ MONDAY: The `state_dict` & Weight Loading
*Task: A pre-trained model is simply a dictionary of strings (layer names) mapped to tensors (matrices of weights). You must learn to inspect and manipulate this dictionary without relying on PyTorch's magic `weights=True` abstraction.*

**Deliverables:**
- [x] **A weight downloading script:** Use `torchvision.models` to download a ResNet-50, but extract *only* its `state_dict` (the raw weights dictionary).
- [x] **A dictionary inspection log:** Write a loop that iterates over the `state_dict` keys. Print the first 10 keys and their corresponding tensor shapes. Notice how the strings directly map to the variable names inside the `nn.Module` classes you built last week.
- [x] **The Mismatch Proof:** Attempt to load these ImageNet weights into a ResNet-50 configured for your 2-class dataset. Catch the exact `RuntimeError` that occurs when PyTorch tries to load a `[1000, 2048]` weight matrix into your `[2, 2048]` classification head.

### 🟥 TUESDAY: Network Surgery & Gradient Freezing
*Task: To fix the shape mismatch from Monday, you must mathematically sever the "Head" of the network and replace it. Then, you must lock the "Body" so the pre-trained ImageNet features are not destroyed by your random initialization.*

**Deliverables:**
- [x] **A surgical replacement script:** Instantiate a pre-trained ResNet-50. Overwrite its final fully connected layer (`model.fc`) with a brand new `nn.Linear` layer sized exactly for your custom dataset.
- [x] **The Freezing Loop:** Iterate through all parameters in the model. Use control flow to set `requires_grad = False` for all layers *except* your newly created `model.fc`.
- [x] **A Parameter Count Audit:** Write a validation function that calculates the number of *trainable* parameters. The console must output a massive difference (e.g., "Total Params: ~23.5M | Trainable Params: ~4.0K"). 

### 🟧 WEDNESDAY: Advanced Augmentation (`albumentations`)
*Task: Standard PyTorch transforms are slow and lack advanced computer vision capabilities (like handling bounding boxes). You will switch to `albumentations`. However, `albumentations` relies on NumPy arrays, not PIL Images.*

**Deliverables:**
- [x] **Library Installation:** Install `albumentations` and `opencv-python`.
- [x] **A standalone YAML augmentation config:** Define a high-intensity pipeline including `RandomResizedCrop`, `HorizontalFlip`, `ColorJitter` (altering brightness/contrast/hue), and `CoarseDropout` (the modern implementation of Cutout, which drops black boxes onto the image).
- [x] **A NumPy transformation script:** Write a script that loads a single image using OpenCV (`cv2.imread`), converts the BGR channel order to RGB, passes it through your `albumentations` pipeline, and plots 5 distinct variations of the image.

### 🟨 THURSDAY: The Dataset Class Refactor
*Task: Your Week 2/Week 5 `Dataset` class is hardcoded to use PIL Images and `torchvision.transforms.v2`. You must rip this out and seamlessly integrate your new `albumentations` pipeline.*

**Deliverables:**
- [x] **A refactored `__getitem__` method:** Your dataset must now read images as NumPy arrays.
- [x] **The Output Conversion:** `albumentations` returns a Python dictionary (e.g., `{"image": augmented_numpy_array}`). You must extract the array, rearrange the dimensions from `[H, W, C]` to PyTorch's required `[C, H, W]` (using `np.transpose` or Albumentation's `ToTensorV2`), and convert it to a `torch.FloatTensor`.
- [x] **A Dataloader Execution:** Pull one batch from your updated `train_loader` to mathematically prove the shapes and types are correct (`[32, 3, 224, 224]`, `torch.float32`).

### 🟩 FRIDAY: Differential Learning Rates (Senior Optimization)
*Task: Freezing the entire body and only training the head is called "Linear Probing." It is safe, but suboptimal. True fine-tuning involves "unfreezing" the top few residual blocks, but training them at a radically slower learning rate than the completely untrained head.*

**Deliverables:**
- [x] **A Layer Unfreezing Script:** Modify your Tuesday script. Keep layers 1, 2, and 3 frozen. Unfreeze Layer 4 and the fully connected Head.
- [x] **Parameter Grouping Integration:** Instead of passing `model.parameters()` to your Optimizer, you must pass a list of dictionaries. 
- [x] **The Differential Optimizer:** Configure the optimizer so that Layer 4 receives a microscopic learning rate (e.g., `1e-5`) to preserve its ImageNet geometry, while the Head receives a standard learning rate (e.g., `1e-3`) so it can learn your specific classes rapidly.

### 🟦 SATURDAY: The Extreme Augmentation Visual Audit
*Task: Cutout and ColorJitter can destroy images. If a black box covers the only identifying feature of a dog, the model is penalized for a wrong guess it couldn't possibly get right. You must visually audit the pipeline.*

**Deliverables:**
- [ ] **A De-normalization & Rendering Script:** Pull a single batch from your `albumentations`-backed `DataLoader`. 
- [ ] **A Sanity Check Grid:** Reverse the mathematical normalization, clip the values to `[0, 1]`, and plot the 32 images in a grid using Matplotlib.
- [ ] **A manual threshold adjustment:** If the `CoarseDropout` is too aggressive (e.g., removing the entire animal), tweak your YAML config parameters until the images are challenging but still human-recognizable. Save the final grid as `augmentation_audit.png`.

### 🟪 SUNDAY: The Transfer Learning Benchmark Race
*Task: Prove visually and mathematically why Transfer Learning runs the modern AI industry. You will race your custom Week 6 architecture against your new pre-trained pipeline.*

**Deliverables:**
- [ ] **Model A Execution:** Train your custom, randomly initialized ResNet-18 (from Week 6) on your dataset for exactly 5 epochs. Log to WandB.
- [ ] **Model B Execution:** Train your new pre-trained, surgically modified, differentially-optimized ResNet-50 on the *exact same dataset* for 5 epochs. Log to WandB.
- [ ] **A WandB Dashboard Export:** Export a single chart showing the Validation Accuracy of both runs. (You should observe the pre-trained model hitting >90% accuracy almost instantly in Epoch 1, while Model A struggles to learn basic edges).
- [ ] **A Git Push** containing your refactored dataset class, surgical training scripts, and `augmentation_audit.png`.