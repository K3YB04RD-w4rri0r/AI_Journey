# 🗓️ WEEK 5: Consolidation, MLOps & Reproducibility
[[Week 4]]
**Pace:** 2 Hours / Day | **Goal:** Transform your messy scripts into a professional, heavily optimized training framework featuring YAML configurations, Mixed Precision training, early stopping, and hyperparameter sweeps.

### ⬛ MONDAY: Absolute Reproducibility (The Seed)
*Task: Deep learning is highly stochastic (random weight initialization, random dataset shuffling, random transforms). If you run a script twice and get two different baseline losses, you cannot scientifically prove an architecture change worked. You must lock down all entropy.*

**Deliverables:**
- [ ] **A global `seed_everything(seed: int)` function** that forces a specific random state across Python's `random`, NumPy's `np.random`, and PyTorch's `torch.manual_seed`.
- [ ] **CUDA determinism flags** explicitly set within that function (research `torch.backends.cudnn.deterministic` and `benchmark`).
- [ ] **A validation script execution** proving reproducibility: Run your initialized (untrained) model on a single validation batch, record the loss. Rerun the entire script. The loss floating-point number must be *exactly* the same down to the 6th decimal place.

### 🟥 TUESDAY: Configuration Management (YAML)
*Task: Hardcoding learning rates into a training script or passing 15 different command-line arguments is an anti-pattern. You must decouple your hyperparameters from your business logic using configuration files.*

**Deliverables:**
- [ ] **A `config.yaml` file** that defines every variable of your pipeline (e.g., `batch_size`, `learning_rate`, `epochs`, `image_size`, `model_depth`, `seed`).
- [ ] **A YAML parsing integration** inside your main training script (using the `PyYAML` or `OmegaConf` libraries).
- [ ] **A complete removal of all hardcoded values** in your training script. Every variable must be dynamically pulled from the parsed YAML dictionary/object.

### 🟧 WEDNESDAY: Hardware Optimization (Mixed Precision Training)
*Task: Standard PyTorch uses 32-bit floats (`float32`). Modern GPUs possess specialized hardware (Tensor Cores) that compute much faster using 16-bit floats (`float16`), halving VRAM usage. You will upgrade your loop to use Automatic Mixed Precision (AMP).*

**Deliverables:**
- [ ] **An integration of `torch.amp.autocast`** wrapping only the forward pass and loss calculation of your training loop (forcing them into `float16` or `bfloat16`).
- [ ] **An integration of `torch.cuda.amp.GradScaler`** (or MPS equivalent, if supported). 16-bit floats can underflow (turn to zero) during backward passes. The scaler mathematically multiplies the gradients to prevent this, then scales them back before the optimizer step.
- [ ] **A benchmarking log** showing the maximum batch size you could fit on your GPU *before* AMP, versus the maximum batch size you can fit *after* AMP.

### 🟨 THURSDAY: Smart Checkpointing & Early Stopping
*Task: Training for exactly 50 epochs and saving the model at the very end is dangerous; the model often overfits at epoch 40, meaning your saved model is worse than an earlier version. You must program the loop to monitor itself.*

**Deliverables:**
- [ ] **An `EarlyStopping` class/mechanism** that tracks the Validation Loss at the end of every epoch.
- [ ] **A `patience` counter** built into the mechanism. If the Validation Loss fails to decrease for `N` consecutive epochs, the script must automatically break the training loop.
- [ ] **A "Best Weights" saving system.** The script must only overwrite `best_model.pth` if the current epoch's Validation Loss is strictly lower than the previous best. 

### 🟩 FRIDAY: Codebase Architecture & Refactoring
*Task: You cannot keep building in a flat directory. You must structure your project exactly like a top-tier open-source repository.*

**Deliverables:**
- [ ] **A standardized directory structure** implemented strictly as follows:
    ```text
    CV_Roadmap/
    ├── configs/            # YAML files
    ├── data/               # Raw and processed datasets (in .gitignore)
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py      # Datasets, Transforms, Collate
    │   ├── model.py        # CNN Architectures, Blocks
    │   ├── engine.py       # Train, Val, EarlyStopping loops
    │   └── utils.py        # Seeding, WandB setup, Metric calculators
    ├── train.py            # The main entry point
    └── inference.py        # The standalone evaluator
    ```
- [ ] **A fully functional `train.py`** that imports cleanly from `src.` modules and executes the end-to-end pipeline using the YAML config.

### 🟦 SATURDAY: Automated Hyperparameter Sweeps
*Task: Finding the perfect Learning Rate and Batch Size manually takes weeks. You will configure Weights & Biases to take over your GPU and automatically search the mathematical space for the best combination.*

**Deliverables:**
- [ ] **A `sweep.yaml` configuration file** defining a search space (e.g., varying learning rates from 1e-5 to 1e-2, batch sizes of 16, 32, 64).
- [ ] **A WandB sweep initialization** (`wandb sweep sweep.yaml`).
- [ ] **An autonomous agent execution** (`wandb agent <SWEEP_ID>`) that takes control of your `train.py` script and runs at least 5 different hyperparameter combinations completely hands-off.
- [ ] **A cloud dashboard review** identifying the single best hyperparameter configuration found by the WandB agent.

### 🟪 SUNDAY: The Portfolio `README` & Final Audit
*Task: A stellar codebase is worthless if nobody knows how to run it. Your final task for Phase 1 is to document your framework so a stranger (or a hiring manager) can replicate your work in seconds.*

**Deliverables:**
- [ ] **A professional `README.md`** written in Markdown. It must include:
    *   A high-level project description.
    *   Hardware requirements.
    *   Step-by-step installation instructions (Conda env creation).
    *   Exact terminal commands to run the training and inference scripts.
- [ ] **A `requirements.txt` or `environment.yml`** file generated so others can identically install your Python dependencies.
- [ ] **A final visual artifact** embedded directly into the `README.md` (e.g., your WandB loss curves or your Confusion Matrix).
- [ ] **A final Git push** concluding Phase 1. 
