# 🎓 MIDTERM EXAM: Weeks 1 & 2
**Total Points:** 100
**Topics:** PyTorch Mechanics, Memory & Strides, Object-Oriented Datasets, Image Processing, & Performance Profiling.

---

## SECTION I: Short Answer & Theory (20 Points)

**1. Memory Mechanics (5 pts)**
You create a tensor `x = torch.arange(12).view(3, 4)`. You then transpose it with `y = x.t()`. Finally, you try to flatten it using `z = y.view(-1)`. 
PyTorch throws a `RuntimeError`. Explain exactly *why* this error occurs at the hardware/memory level, and state the two different ways you can fix the code.

**2. Broadcasting (5 pts)**
You have a batch of images with the shape `batch =[64, 3, 256, 256]`. You have a mean tensor for your color channels `means = torch.tensor([0.485, 0.456, 0.406])` which has a shape of `[3]`. 
If you try to execute `batch - means`, what will happen? If it fails, how exactly must you reshape `means` to make the subtraction work via broadcasting?

**3. Data Leakage (5 pts)**
On Week 2, Monday, you hashed image files to find duplicates *before* creating your `train.csv` and `val.csv` manifests. Conceptually, what is "data leakage" in this context, and why is doing deduplication *after* your train/val split dangerous?

**4. Dataloader Profiling (5 pts)**
You benchmarked your DataLoader on Sunday. Explain the difference in execution between setting `num_workers=0` versus `num_workers=4` in your `DataLoader`. What is the CPU doing differently?

---

## SECTION II: Spot the Bug (20 Points)
*Each of the following code snippets contains a critical error based on the concepts you learned. Identify the bug and write the corrected line(s) of code.*

**Bug 1: The Visualizer (5 pts)**
```python
# Goal: Plot the first image in the batch
batch = torch.randn(32, 3, 224, 224)
img = batch[0] # Shape is [3, 224, 224]
plt.imshow(img.cpu().numpy())
plt.show()
```

**Bug 2: The Manual Activation (5 pts)**
```python
# Goal: Manually apply a ReLU activation (set all negative values to 0)
x = torch.randn(10, 10)
x[x > 0] = 0
```

**Bug 3: The Custom Dataset (5 pts)**
```python
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __getitem__(self, idx):
        # Goal: Return the image path and label
        row = self.data.iloc[idx]
        return row['image_path'], row['label']
```
*(Hint: Look closely at what PyTorch `Dataset` classes are expected to return and what is missing here).*

**Bug 4: The Hardware Transfer (5 pts)**
```python
# Goal: Move a tensor to the GPU and multiply it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000, 1000)
x.to(device)
y = x @ x
```

---

## SECTION III: Applied Implementation (40 Points)

**1. The `collate_fn` (15 pts)**
Write a custom `collate_fn` function in pure Python/PyTorch. 
*   **Input:** A list of tuples containing `(image_tensor, label_int)` of length `B` (Batch size).
*   **Output:** A single batched image tensor of shape `[B, C, H, W]` and a single batched label tensor of shape `[B]`.
```python
def custom_collate(batch_list):
    # YOUR CODE HERE
    pass
```

**2. Streaming Statistics (15 pts)**
You are calculating the running Mean and Standard Deviation of a massive dataset. 
You have tracked three variables over your loop: 
*   `total_pixels` (an integer)
*   `sum_pixels` (a tensor of shape `[3]` representing the sum of R, G, B values)
*   `sum_sq_pixels` (a tensor of shape `[3]` representing the sum of squared R, G, B values).
Write the mathematical formulas (in Python code) to calculate the final `mean` and `std` tensors from these three variables.

**3. CLI Engineering (10 pts)**
Write the Python boilerplate to set up `argparse`. Make a parser that accepts a `--batch_size` (integer, default 32) and a `--device` (string, default "cpu"), and print them out.

---

## SECTION IV: Architecture & Design (20 Points)

**1. The "Pad to Square" Transformation (10 pts)**
Standard resizing squishes rectangular images. Explain the algorithmic logic you used to create your `PadToSquare` transform. 
*   How do you determine the dimensions of the black background? 
*   How do you calculate where to paste the original image so that it is perfectly centered?

**2. De-normalization (10 pts)**
At the end of your pipeline, you had a tensor that was normalized using `(image - mean) / std`. 
*   What is the exact algebraic formula to reverse this normalization?
*   Why do normalized images look "dark and distorted" if you try to plot them without reversing the normalization? 

***

### 🏁 Instructions for Submission
Take your time and answer these based purely on what you absorbed over the last 14 days. Do not look at your old code! 