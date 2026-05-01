# 🗓️ WEEK 2: Bulletproof Data Pipelines

**Pace:** 2 Hours / Day | **Goal:** Build a production-grade data pipeline that handles corrupted files, maintains aspect ratios, and feeds the GPU at maximum speed.

### ⬛ MONDAY: Data Janitor Work & Leakage Prevention

_Task: Acquire real-world data, sanitize it by removing corrupted files, eliminate hidden duplicates (which cause data leakage), and generate training manifests._

**Deliverables:**

- [x] **A raw dataset folder** containing at least 1,000 images divided into two distinct classes (e.g., Cats vs. Dogs).
- [x] **A cleansing script execution** that iterates through the raw folder, attempts to read every image header, catches any read exceptions, and deletes corrupted/hidden OS files.
- [x] **A deduplication script execution** that reads every image file as raw bytes, calculates its cryptographic hash (e.g., MD5), identifies exact duplicates, and deletes them. Output a print statement of how many duplicates were found.
- [x] **Two CSV files** (`train.csv` and `val.csv`) mapping the surviving file paths to integer labels (0 or 1), representing an 80/20 randomized split.

### 🟥 TUESDAY: Object-Oriented PyTorch `Dataset`

_Task: Build the foundational PyTorch class that reads from your CSV manifests and safely loads individual images into memory in a standardized format._

**Deliverables:**

- [x] **A `dataset.py` file** containing a custom class that strictly inherits from `torch.utils.data.Dataset`.
- [x] **An `__init__` method** that loads your CSV manifests into memory.
- [x] **A `__len__` method** that accurately returns the total number of items in the manifest.
- [x] **A `__getitem__` method** that reads the image from disk, standardizes it to exactly 3 color channels (handling any grayscale or RGBA anomalies natively), applies optional transforms, and returns a tuple of `(image, label)`.
- [x] **A test execution** proving that fetching index `[0]` returns a valid image object and the correct integer label.

### 🟧 WEDNESDAY: Geometric Transformations

_Task: Standard resizing distorts images by squishing them. You must build a custom transformation that pads an image with black bars to make it square before resizing, preserving its true aspect ratio._

**Deliverables:**

- [x] **A custom Python class** (e.g., `PadToSquare`) built to be used as a transform. It must dynamically find an image's longest dimension and paste the image into the dead center of a black square of that same dimension.
- [x] **A transform pipeline** utilizing modern PyTorch transforms (`v2.Compose`) that applies your `PadToSquare` class, converts the image to a tensor, scales the pixel values to a standard float range, and resizes the square to exactly `224x224`.
- [x] **A visual test** proving your pipeline successfully takes a highly rectangular image and returns a `224x224` tensor with black bars and zero distortion.

### 🟨 THURSDAY: Memory-Efficient Dataset Statistics

_Task: To normalize data, you need the Mean and Standard Deviation of your specific dataset. Because the dataset exceeds available RAM, you must compute these statistics iteratively using running sums._

**Deliverables:**

- [ ] **A temporary iterator/loader** set up to loop through your entire training set in batches (with resizing applied, but no normalization yet).
- [ ] **An accumulation script** that calculates the running sum of pixels and the running sum of _squared_ pixels for the Red, Green, and Blue channels independently as it loops through the batches.
- [ ] **A final calculation step** that uses the accumulated sums and the total pixel count to compute the exact `[R, G, B]` Mean and Standard Deviation. Print these 6 floats to the console.
- [ ] **An updated transform pipeline** that includes a normalization step explicitly utilizing your newly calculated statistics.

### 🟩 FRIDAY: Custom Batching (`collate_fn`) & DataLoaders

_Task: Take control of how PyTorch batches individual items together. Write the manual assembly logic to stack images and labels, then optimize the hardware loading._

**Deliverables:**

- [ ] **A custom collate function** that accepts a list of tuples (the output of `__getitem__`), separates the images from the labels, and constructs a single batched image tensor and a single batched label tensor.
- [ ] **A `train_loader` and `val_loader`** instantiated using your custom dataset, your custom collate function, and specific arguments to enable multiprocessing and fast CPU-to-GPU memory transfers.
- [ ] **A test execution** that successfully draws exactly one batch from the loader and prints the resulting tensor shapes.

### 🟦 SATURDAY: The Visual Sanity Check

_Task: Normalized tensors look dark and distorted when plotted directly. You must mathematically reverse your normalization to visually prove the pipeline is feeding the correct data and labels._

**Deliverables:**

- [ ] **A "de-normalization" function** that mathematically reverses the normalization applied on Thursday and bounds the values safely for rendering.
- [ ] **A batch extraction** that pulls a single batch from your active `train_loader`.
- [ ] **A single saved image file** (`pipeline_proof.png`) containing a visual grid of the batch. The grid must display properly colored images (not distorted by aspect ratio squishing), and each image must have its corresponding ground truth label written above it as a title.

### 🟪 SUNDAY: Profiling Bottlenecks & Delivery

_Task: Measure your pipeline's speed to ensure the CPU can load images fast enough to keep a future GPU busy, clean up your codebase, and push your portfolio artifact._

**Deliverables:**

- [ ] **A benchmarking script** that loops through 50 batches of your dataloader without doing any model training, recording the total execution time.
- [ ] **A documented experiment** inside your script (or in comments) logging the total execution time when your dataloader uses 0 multiprocessing workers versus multiple workers.
- [ ] **A fully linted codebase** with zero formatting or unused-import warnings (using your linter of choice).
- [ ] **A Git push** containing your clean `.py` files, your CSV manifests, and your `pipeline_proof.png` (ensuring no raw image data is pushed to the repository).