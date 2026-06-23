
# 🗓️ WEEK 8: The Accuracy Paradox & Imbalanced Topologies
**Pace:** 2 Hours / Day | **Goal:** Synthesize a heavily skewed dataset, implement inverse-frequency loss weighting, engineer dataloader-level oversampling, and build a custom Focal Loss function.

### ⬛ MONDAY: Synthetic Skew & The Baseline of Failure
*Task: You must deliberately corrupt your clean CSV manifest to create a 95% to 5% class imbalance. Then, you will train a baseline model to empirically observe the "Accuracy Paradox."*

**Deliverables:**
- [ ] **A dataset corruption script:** Write a script that loads your `train.csv`. Identify the "Class 1" rows and randomly delete 90% of them. Save this as `train_imbalanced.csv`. Leave `val.csv` perfectly balanced (so evaluation remains fair).
- [ ] **A baseline training execution:** Train your pre-trained ResNet pipeline from Week 7 on this corrupted dataset for 5 epochs without changing any logic.
- [ ] **An evaluation log:** Print the Validation Accuracy, Validation Precision (Class 1), and Validation Recall (Class 1). *Note: You should observe an overall Validation Accuracy of around 90-95%, but a Class 1 Recall of 0% (the model is just guessing Class 0 every time).*

### 🟥 TUESDAY: Multi-Class Metric Accumulators
*Task: You can no longer rely on overall "Accuracy." You must build a tracking engine that calculates True Positives, False Positives, and False Negatives for every class dynamically at the end of each validation epoch.*

**Deliverables:**
- [ ] **A Custom `MetricTracker` Class:** This class must ingest raw network logits and ground truth labels batch-by-batch, computing the confusion matrix iteratively without storing thousands of tensors in RAM.
- [ ] **Mathematical Class Methods:** Implement methods inside the class to compute Macro-F1 (the unweighted mean of F1 across classes) and Micro-F1. 
- [ ] **WandB Integration:** Wire this tracker into your validation loop. Ensure your Weights & Biases dashboard now displays separate line charts for `Recall_Class_0` and `Recall_Class_1`.

### 🟧 WEDNESDAY: Algorithmic Mitigation (Weighted Cross-Entropy)
*Task: The easiest way to fix imbalance is to change the loss landscape. You will calculate the inverse frequency of your classes and apply a mathematical penalty multiplier to the network when it misses the rare class.*

**Deliverables:**
- [ ] **A mathematical weighting function:** Write a script that iterates through `train_imbalanced.csv`, counts the exact occurrences of Class 0 and Class 1, and calculates their inverse frequency weights. (e.g., $W_c = \frac{\text{Total Samples}}{C \times \text{Samples in Class C}}$).
- [ ] **A Loss Function replacement:** Instantiate `nn.CrossEntropyLoss` but pass your calculated weights to the `weight` parameter. Ensure the weight tensor is moved to your target device (CUDA/MPS).
- [ ] **A gradient impact proof:** Run exactly one batch containing both classes. Print the raw scalar loss. Now run it with the standard, unweighted loss. Print the scalar loss. Document the numerical difference the weights caused.

### 🟨 THURSDAY: Data-Level Mitigation (The `WeightedRandomSampler`)
*Task: Modifying the loss function can cause unstable gradients. The industry-preferred alternative is to fix the problem at the `DataLoader` level by mathematically over-sampling the rare class so the GPU sees a 50/50 split in every batch.*

**Deliverables:**
- [ ] **A sample-weight mapping:** Instead of calculating two weights for the *classes*, you must calculate a specific probability weight for *every single image* in your dataset based on its class, resulting in a list/tensor of weights matching your dataset length.
- [ ] **The Sampler integration:** Instantiate PyTorch's `WeightedRandomSampler` using your sample weights. 
- [ ] **The Dataloader configuration:** Pass the sampler to your `train_loader`. *Crucial constraint: You must read the documentation to figure out which standard `DataLoader` argument strictly conflicts with a custom sampler and disable it to prevent a runtime crash.*
- [ ] **A batch verification script:** Pull a single batch from the new loader. Print the class counts inside that specific batch. It should be roughly 50/50, despite the underlying dataset being 95/5.

### 🟩 FRIDAY: Architectural Mitigation (Custom Focal Loss)
*Task: Standard Cross-Entropy heavily penalizes a model when it is wrong, but it still applies a small penalty when it is right. In a skewed dataset, millions of "small penalties" from the majority class overwhelm the gradients. You will implement the mathematics of Focal Loss (Lin et al., 2017).*

**Deliverables:**
- [ ] **A custom `FocalLoss` Module:** Inherit from `nn.Module`. 
- [ ] **The Mathematical Implementation:** Implement the formula: $FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$. You must extract the predicted probabilities using `Softmax` (or `LogSoftmax`), apply the focusing parameter ($\gamma$), and calculate the final scalar.
- [ ] **A numerical edge-case handler:** Ensure you apply a tiny epsilon (e.g., `1e-7`) to your probabilities before taking the log to prevent `NaN` (Not a Number) explosions in your gradients.

### 🟦 SATURDAY: The Mitigation Bake-Off
*Task: You have engineered three distinct solutions to the imbalance problem. You will now run a controlled experiment to see which one performs best on your specific dataset.*

**Deliverables:**
- [ ] **Experiment 1 Execution:** Train for 5 epochs using Weighted Cross-Entropy.
- [ ] **Experiment 2 Execution:** Train for 5 epochs using the `WeightedRandomSampler`.
- [ ] **Experiment 3 Execution:** Train for 5 epochs using your custom `FocalLoss`.
- [ ] **A comparative dashboard analysis:** Group the three runs in WandB. Identify which method yielded the highest Macro-F1 score without destroying the precision of the majority class.

### 🟪 SUNDAY: YAML Strategy Toggles & Delivery
*Task: Hardcoding samplers or loss functions breaks reproducibility. You must refactor your codebase so you can toggle these mitigation strategies instantly from the terminal or config file.*

**Deliverables:**
- [ ] **YAML Config updates:** Add an `imbalance_strategy` field to your `config.yaml` that accepts string values like `"none"`, `"weighted_loss"`, `"sampler"`, or `"focal"`.
- [ ] **Engine refactoring:** Modify your `train.py` script to read this YAML string and dynamically instantiate the correct loss function or dataloader.
- [ ] **A fully linted codebase** checking for unused imports.
- [ ] **A Git push** containing your custom `MetricTracker`, your `FocalLoss` module, and a WandB export image proving your model can now detect the 5% minority class.