# 🗓️ WEEK 10: Object Detection Math – IoU & NMS
**Pace:** 2 Hours / Day | **Goal:** Parse the YOLOv1 architecture, master bounding box coordinate translations, vectorize Intersection over Union natively on the GPU, and build a multi-class greedy NMS algorithm.

### ⬛ MONDAY: Literature Review & Coordinate Formats
*Task: You must read Redmon et al.'s YOLOv1 paper. Then, you must build the foundational functions to translate bounding boxes between the two dominant mathematical formats: Center-based (used by the model) and Corner-based (used for rendering/IoU).*

**Deliverables:**
- [x] **A read and annotated PDF** of the YOLOv1 paper. Pay specific attention to the $S \times S$ grid, the $B$ bounding boxes, the confidence score formula ($Pr(Object) * IOU^{truth}_{pred}$), and the final output tensor shape ($S \times S \times (B * 5 + C)$).
- [x] **A `box_cxcywh_to_xyxy` function:** Write a PyTorch function that takes a tensor of boxes formatted as `[center_x, center_y, width, height]` and mathematically transforms them into `[x_min, y_min, x_max, y_max]`. 
- [x] **A `box_xyxy_to_cxcywh` function:** Write the exact mathematical inverse.
- [x] **An assertion test script:** Generate a random tensor of 100 boxes in `cxcywh` format. Run it through the first function, then the second. Assert mathematically that the final tensor is identical to the starting tensor (handling floating-point precision tolerances).

### 🟥 TUESDAY: Intersection over Union (The Core Metric)
*Task: IoU is the bedrock of all Object Detection. It measures how much two bounding boxes overlap. You will build this strictly using PyTorch operations (`torch.max`, `torch.min`, `torch.clamp`).*

**Deliverables:**
- [x] **An `intersection_area` calculation:** Given two boxes in `xyxy` format, calculate the coordinates of the intersecting rectangle. Use PyTorch clamping to ensure that if the boxes do NOT overlap, the width or height of the intersection evaluates to exactly `0.0`.
- [x] **A `union_area` calculation:** Mathematically, the union is `Area(Box1) + Area(Box2) - Intersection_Area`.
- [x] **An `iou_single` function:** Return the ratio of Intersection over Union. Include defensive programming to add a tiny epsilon (e.g., `1e-6`) to the denominator to mathematically prevent "Divide by Zero" explosions.
- [x] **A Console Proof:** Run your function on two perfectly overlapping boxes (must print `1.0`), two completely separated boxes (must print `0.0`), and a box that overlaps exactly half of another (must print `0.5`).

### 🟧 WEDNESDAY: Batched Tensor Operations (Senior Vectorization)
*Task: A naive `iou_single` function using a `for` loop to compare 1,000 predictions against 1,000 ground-truth boxes will completely bottleneck your GPU. You must rewrite IoU using PyTorch broadcasting to compute an entire grid of overlaps simultaneously.*

**Deliverables:**
- [x] **A `batched_iou` function:** This function must accept a tensor of shape `[N, 4]` and a tensor of shape `[M, 4]`. Using PyTorch unsqueezing and broadcasting, it must return an `[N, M]` matrix containing every single IoU combination simultaneously, without a single `for` loop.
- [x] **A CPU vs GPU Vectorization Benchmark:** Generate two sets of 5,000 random boxes. 
    1. Time how long it takes to calculate all combinations using a nested Python `for` loop.
    2. Time how long your `batched_iou` takes on the CPU.
    3. Time how long your `batched_iou` takes on your CUDA/MPS device.
- [x] **A log of the benchmark results**, explicitly proving the massive speed multiplier of vectorization.

### 🟨 THURSDAY: The Greedy Algorithm (Single-Class NMS)
*Task: YOLO predicts multiple boxes for the same object. NMS removes duplicates by sorting boxes by confidence score, keeping the highest, and deleting any remaining box that overlaps it beyond a certain IoU threshold.*

**Deliverables:**
- [x] **A `single_class_nms` function:** Accepts three arguments: `boxes` (shape `[N, 4]`), `scores` (shape `[N]`), and `iou_threshold` (float).
- [x] **A sorting mechanism:** Use `torch.argsort` in descending order to arrange the boxes by their confidence scores.
- [x] **The Greedy Loop:** Implement a `while` loop that:
    1. Pops the highest-scoring box and saves its index.
    2. Uses your `batched_iou` to compare this box against all remaining boxes.
    3. Creates a boolean mask to filter out (suppress) any box where the IoU exceeds your `iou_threshold`.
- [x] **An output tensor** returning the indices of the boxes that survived suppression.

### 🟩 FRIDAY: Multi-Class NMS (The Offset Trick)
*Task: Real images contain overlapping objects of *different* classes (e.g., a person holding a dog). Standard NMS might accidentally suppress the dog's box because the person's highly-confident box overlaps it. You must upgrade NMS to be class-aware.*

**Deliverables:**
- [x] **A `multi_class_nms` function:** Add a `classes` (shape `[N]`) argument to your function.
- [x] **The "Offset Trick" Implementation:** Instead of running a slow `for` loop over every individual class, multiply the `classes` tensor by a massive scalar (e.g., `4096.0`) and add this offset to the bounding box coordinates. *Why? This mathematically physically separates different classes by thousands of pixels in space, guaranteeing their IoU will evaluate to `0.0` while keeping same-class boxes intact.*
- [x] **The Vectorized Execution:** Pass these massively offset boxes into your standard `single_class_nms` logic.
- [x] **A test execution** proving that a highly overlapping "Dog" box and "Person" box both survive, while a duplicate "Dog" box is suppressed.

### 🟦 SATURDAY: The Visual Simulation Bench
*Task: Bounding boxes are abstract numbers. You must build a visual testbench using OpenCV or Matplotlib to prove your algorithm cleans up "raw" network outputs.*

**Deliverables:**
- [ ] **A raw output simulator:** Write a script that generates a 512x512 blank image. Mathematically generate 1 "Ground Truth" box in the center. Then, generate 20 "Predicted" boxes by adding random gaussian noise to the center coordinates and dimensions. Assign random confidence scores to all 20.
- [ ] **A rendering function:** Write a function that iterates through a tensor of boxes and draws rectangles on the image, adding the confidence score as text above the box.
- [ ] **The Visual Proof:** Render two images side-by-side using Matplotlib. The Left Image must show all 20 messy overlapping boxes (Before NMS). The Right Image must run your `multi_class_nms` function and plot *only* the surviving boxes (After NMS). Save as `nms_simulation.png`.

### 🟪 SUNDAY: Edge Cases, Hardening & Delivery
*Task: Your math works in a vacuum, but in a production pipeline, tensors do weird things. You must harden the code against edge cases that crash training loops.*

**Deliverables:**
- [ ] **Defensive Checks for Empty Tensors:** Add logic to your NMS function to immediately return an empty tensor if the input `boxes` tensor is size `0`. (This happens frequently if a network detects nothing in an image).
- [ ] **Defensive Checks for Malformed Boxes:** Add a check to drop any bounding box where `x2 <= x1` or `y2 <= y1` (an impossible area of zero or negative space).
- [ ] **Code Formatting:** Run your Ruff linter over `iou.py` and `nms.py`. Ensure type-hinting (e.g., `boxes: torch.Tensor`) is rigorous.
- [ ] **A Git Push** containing your cleanly separated math modules, your benchmark test script, and the `nms_simulation.png` artifact.