# 🗓️ WEEK 6: Paper to Code – The ResNet Architecture
**Pace:** 2 Hours / Day | **Goal:** Understand the degradation problem, master 1x1 convolutions, implement mathematical identity mappings (skip connections), and dynamically assemble a custom ResNet-18.

### ⬛ MONDAY: Literature Review & The Degradation Problem
*Task: You must read the original "Deep Residual Learning for Image Recognition" paper. Then, you will empirically prove the problem the authors were trying to solve: that simply stacking more layers eventually ruins a network.*

**Deliverables:**
- [x] **A read and annotated PDF** of the ResNet paper. Pay specific attention to *Figure 1* (The Degradation Problem), *Equation 1* ($y = F(x) + x$), and *Table 1* (Architectures).
- [x] **A "Plain Network" Generator Script:** Write a script that dynamically generates a CNN with $N$ basic convolutional layers (no skip connections). 
- [ ] **An Empirical Proof:** Initialize a 10-layer Plain Network and a 30-layer Plain Network. Push a single batch of images through both and trigger a backward pass. Print the mean of the gradients for the first layer of both models. You must observe the 30-layer network's gradients approaching zero (The Vanishing Gradient).

### 🟥 TUESDAY: The Basic Residual Block ($F(x) + x$)
*Task: Implement the core mathematical innovation of the paper. A Residual Block allows gradients to bypass non-linearities and flow completely unimpeded through the network via an "Identity Shortcut."*

**Deliverables:**
- [x] **A custom `BasicBlock` module** (`nn.Module`). It must contain exactly: Conv3x3 $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Conv3x3 $\rightarrow$ BatchNorm.
- [x] **The Skip Connection Math:** In the `forward` pass, you must literally write the Python addition of the original input tensor to the output of the second BatchNorm, *before* applying the final ReLU.
- [x] **A Shape Sanity Check:** Pass a `[1, 64, 56, 56]` tensor through your initialized block. The script must assert that the output shape is mathematically identical to the input shape.

### 🟧 WEDNESDAY: The Projection Shortcut (1x1 Convolutions)
*Task: You cannot add tensor $A$ to tensor $B$ if they are different shapes. As a network gets deeper, spatial dimensions halve (via `stride=2`) and channel dimensions double. You must implement a "Projection Shortcut" to mathematically force the shapes to match.*

**Deliverables:**
- [x] **A modified `BasicBlock`** that accepts a `stride` argument and an `expansion` factor.
- [x] **A 1x1 Convolution implementation:** If the input shape will not match the output shape, your block must dynamically generate a `shortcut` sequential layer consisting of a `1x1` Convolution (with a matching stride) and a BatchNorm.
- [x] **A Projection Sanity Check:** Pass a `[1, 64, 56, 56]` tensor into a block configured with `stride=2` and `out_channels=128`. The script must execute without shape-mismatch crashes and output a `[1, 128, 28, 28]` tensor.

### 🟨 THURSDAY: The Bottleneck Block (For ResNet-50+)
*Task: Deep ResNets (50, 101, 152 layers) cannot afford to use two 3x3 convolutions per block—it requires too much VRAM. You will implement the "Bottleneck" mathematical trick: shrink the channels, do the 3x3 math, then expand the channels.*

**Deliverables:**
- [x] **A custom `BottleneckBlock` module** (`nn.Module`). 
- [x] **The 1-3-1 Architecture:** It must contain: Conv1x1 (reduce channels) $\rightarrow$ Conv3x3 $\rightarrow$ Conv1x1 (expand channels by a factor of 4).
- [x] **A Parameter Count Proof:** Write a script that compares the total trainable parameters of your `BasicBlock` vs your `BottleneckBlock` given the exact same input/output channels. Print the mathematical VRAM savings.

### 🟩 FRIDAY: Dynamic ResNet Assembly (The `_make_layer` Engine)
*Task: Hardcoding 18 or 50 blocks is an amateur mistake. You must write a dynamic constructor that builds the network mathematically based on a list of block counts (e.g., `[2, 2, 2, 2]` for ResNet-18).*

**Deliverables:**
- [x] **A `CustomResNet` module.** 
- [x] **The Stem implementation:** The required initial `Conv7x7` (stride 2), BatchNorm, ReLU, and `MaxPool3x3` (stride 2) that reduces the initial image size before the residual blocks begin.
- [x] **A `_make_layer` helper method** inside the class. It must take a block type (`BasicBlock` or `BottleneckBlock`), a channel count, a number of blocks, and a stride. It must loop to generate the sequence, ensuring only the *first* block in the sequence handles the stride/projection downsampling.
- [x] **The Head implementation:** The final Global Average Pooling (`nn.AdaptiveAvgPool2d(1)`) and the fully connected Linear layer matching your dataset's class count.

### 🟦 SATURDAY: Pipeline Integration & Parameter Verification
*Task: Prove your hand-written architecture aligns perfectly with the multi-million dollar models built by Kaiming He's team at Microsoft, and wire it into your Week 5 YAML pipeline.*

**Deliverables:**
- [x] **A conceptual verification script:** Instantiate your `CustomResNet` configured identically to a standard ResNet-18. Calculate your exact total trainable parameters. It should perfectly match the official ResNet-18 count (roughly 11.1 to 11.2 Million, depending on the final FC layer).
- [ ] **Pipeline wiring:** Add `"resnet18"` as a valid `model_architecture` option in your Week 5 `config.yaml`.
- [x] **A Dry-Run Execution:** Run your `train.py` script from Phase 1 using your new ResNet model for exactly 1 epoch using your dataset. Ensure there are no CUDA Out Of Memory (OOM) errors and no shape mismatch crashes during the backward pass.

### 🟪 SUNDAY: The "Degradation Defeated" Visual Proof
*Task: You read the paper on Monday. On Sunday, you replicate its findings. You will train a Plain network and a Residual network side-by-side to prove your skip connections actually work.*

**Deliverables:**
- [x] **A localized training experiment:** Initialize a deep Plain Network (e.g., 18 layers, no skip connections) and your `CustomResNet-18`. 
- [x] **A dual WandB tracking run:** Train both models on your dataset for 10 epochs using identical hyperparameters (learning rate, batch size, seed). Log both runs to your Weights & Biases dashboard.
- [x] **A final exported graph (`degradation_proof.png`)** from WandB showing the Training Loss curves of both models on the same chart. The ResNet loss must fall faster and lower than the Plain network. 
- [x] **A Git Push** containing your modular `resnet.py` and your visual proof.