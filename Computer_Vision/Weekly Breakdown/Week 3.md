# 🗓️ WEEK 3: The Engine – Autograd, Optimizers & The Training Loop
**Pace:** 2 Hours / Day | **Goal:** Master computational graphs, build a bare-minimum model, construct a mathematically sound training/validation loop, and prove it works by deliberately overfitting a single batch.

### ⬛ MONDAY: Autograd & The Computational Graph
*Task: Before using built-in optimizers, you must understand how PyTorch tracks calculus via the computational graph. You will build a synthetic mathematical function and optimize its parameters manually.*

**Deliverables:**
- [ ] **A script generating a synthetic dataset** of inputs $X$ and targets $Y$ based on a simple linear equation (e.g., $Y = 3X + 2$).
- [ ] **Weight and Bias tensors** initialized randomly, explicitly configured to track gradients (`requires_grad=True`).
- [ ] **A manual gradient descent loop** (no `torch.optim`, no `torch.nn`). You must calculate the forward pass, calculate the Mean Squared Error loss, trigger the backward pass, and update the weights manually.
- [ ] **A context manager implementation** proving you know how to safely update weights without PyTorch adding the update step itself to the computational graph (Hint: investigate `torch.no_grad()`).
- [ ] **Console output** showing your random weights gradually converging toward the true values (e.g., $W \approx 3$, $b \approx 2$).

### 🟥 TUESDAY: The Dummy Model & Loss Functions
*Task: We are not building complex CNNs yet (that is Week 4). Today, you need a structurally correct, bare-minimum PyTorch model to plug your image data into, and an objective function to measure its failure.*

**Deliverables:**
- [ ] **A custom class** inheriting from `torch.nn.Module`.
- [ ] **An `__init__` method** defining a flattening layer and a single fully connected layer (`nn.Linear`). The input features must mathematically match the total pixels of a single image from your Week 2 dataloader. The output features must equal your number of classes (2).
- [ ] **A `forward` method** that defines how data flows through the layers.
- [ ] **An instantiation** of your model moved to your target hardware device (CUDA/MPS).
- [ ] **A script execution** that pulls exactly one batch from your Week 2 `train_loader`, passes it through the model, and computes the Cross Entropy Loss. Print the initial scalar loss value to the console.

### 🟧 WEDNESDAY: The Optimizer & Core Loop Mechanics
*Task: Replace Monday's manual gradient math with PyTorch's native optimization algorithms. Master the 5 strict sequential steps required to train a neural network.*

**Deliverables:**
- [ ] **An optimizer instantiation** (e.g., `torch.optim.Adam` or `SGD`) linked to your dummy model's parameters, with a defined learning rate.
- [ ] **A single-step training function** that executes the unchangeable 5-step lifecycle:
    1. Forward pass.
    2. Loss calculation.
    3. Gradient clearing.
    4. Backpropagation (calculating derivatives).
    5. Optimizer step (updating weights).
- [ ] **A diagnostic printout** that proves weights are changing. Extract a specific weight value from the model *before* the 5-step process, and print it again *after* the process to visually verify the mathematical update occurred.

### 🟨 THURSDAY: The Ultimate Sanity Check (Overfit a Single Batch)
*Task: This is the most important debugging technique in Deep Learning. If a model and training loop cannot memorize a single batch of data perfectly, the architecture or pipeline is fundamentally broken. You will force your model to memorize 32 images.*

**Deliverables:**
- [ ] **A batch isolation script** that extracts exactly one batch of images and labels from your `train_loader` and traps them in memory (do not iterate the loader).
- [ ] **A micro-training loop** that runs your 5-step optimization process on *this exact same batch* for 100 consecutive iterations.
- [ ] **Accuracy tracking logic** inside the loop that converts raw model logits into predicted class indices and calculates the percentage of correct predictions.
- [ ] **A plotted curve** saved as `single_batch_overfit.png` visually proving that within 100 iterations, the Loss converged to ~0.000 and the Accuracy reached exactly 100%.

### 🟩 FRIDAY: Full Epoch Training & Running Metrics
*Task: Scale the loop to ingest the entire dataset. You must correctly accumulate running metrics. (Warning: A common junior error is calculating running averages incorrectly because the final batch in a dataset is often smaller than the rest).*

**Deliverables:**
- [ ] **Nested loops** implemented cleanly: an outer loop for `Epochs` and an inner loop for `Batches` (iterating over the entire `train_loader`).
- [ ] **Correct running accumulators** that track total cumulative loss and total correct predictions across the epoch.
- [ ] **Math-safe metric calculations** that divide cumulative metrics by the actual number of *samples* processed, not just the number of batches.
- [ ] **Console output** for an entire 3-epoch training run, displaying the exact Training Loss and Training Accuracy at the end of each epoch.

### 🟦 SATURDAY: State Management & The Validation Phase
*Task: A model that only trains is useless; it must be evaluated on unseen data. You must carefully manage PyTorch's internal state to ensure you do not accidentally train on the validation set or leak gradients.*

**Deliverables:**
- [ ] **State toggles** added to your loop. The model must explicitly be set to training mode before the train loop, and evaluation mode before the validation loop.
- [ ] **A complete validation loop** nested inside the epoch loop (running after the training phase completes).
- [ ] **Gradient disabling** applied strictly to the validation loop using a context manager, proving you understand how to prevent memory leaks and unnecessary calculus during inference.
- [ ] **Combined console output** logging Train Loss, Train Acc, Validation Loss, and Validation Acc side-by-side for 5 total epochs.

### 🟪 SUNDAY: Checkpointing & Artifact Delivery
*Task: Training a model costs compute and time. If the script ends, the weights vanish from RAM. You must serialize the mathematical state to your hard drive and clean up the repository.*

**Deliverables:**
- [ ] **A checkpointing mechanism** that saves the model's `state_dict` to disk as a `.pth` or `.pt` file at the very end of training.
- [ ] **A "load and verify" script** completely separate from your training script. It must initialize a fresh, untrained model, load the `.pth` weights into it from the hard drive, and run a single validation batch to prove the loaded weights produce the exact same predictions as the end of your training run.
- [ ] **A fully linted codebase** with zero formatting or unused-import warnings.
- [ ] **A Git push** containing your training scripts, your model definition, and your `single_batch_overfit.png`. *(Do NOT commit the `.pth` file to version control. Add `*.pth` to your `.gitignore`)*.