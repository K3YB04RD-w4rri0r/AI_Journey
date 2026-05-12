# 🗓️ WEEK 4: Convolutions, Cloud Tracking & Rigorous Evaluation
**Pace:** 2 Hours / Day | **Goal:** Master the spatial mathematics of CNNs, build a modular architecture, track system state in Weights & Biases (WandB), and evaluate performance using multi-dimensional metrics.

### ⬛ MONDAY: Convolution Mechanics & Reusable Blocks
*Task: You must understand exactly how a convolution operation reduces or maintains spatial dimensions. You will write the mathematical formula in code to dynamically calculate shapes, and build a reusable layer block.*

**Deliverables:**
- [ ] **A spatial math function** that accepts input size, kernel size, padding, and stride, and returns the exact Output Height/Width mathematically. *(Hint: Research the `(W - F + 2P) / S + 1` formula).*
- [ ] **A `ConvBlock` custom module** (`nn.Module`). Instead of writing out every layer individually, this class should encapsulate a Convolution -> Batch Normalization -> ReLU Activation -> Max Pooling sequence.
- [ ] **A test execution** passing a dummy `[1, 3, 224, 224]` tensor into your `ConvBlock`. Print the output shape and verify it matches your mathematical function's prediction.

### 🟥 TUESDAY: The Modular CNN Architecture
*Task: Assemble your reusable blocks into a complete Convolutional Neural Network. You must calculate exactly how many features enter your final fully connected (Linear) layer without relying on trial-and-error.*

**Deliverables:**
- [ ] **A `CustomCNN` class** that stacks 3 to 4 of your `ConvBlocks`, progressively increasing the channel depth (e.g., 32 -> 64 -> 128) while decreasing the spatial dimensions.
- [ ] **A mathematically sound `nn.Linear` classifier head** at the end of the network. You must use your math from Monday to calculate the exact `in_features` of this linear layer based on a `224x224` input image passing through your specific poolings.
- [ ] **A parameter counting script** that iterates over `model.parameters()`, calculates the total number of trainable weights in your network, and prints it to the console (e.g., "Total Trainable Params: 1.2M").

### 🟧 WEDNESDAY: Cloud Experiment Tracking (WandB Init)
*Task: Local print statements are useless for long-running experiments. You will integrate Weights & Biases to track your hyperparameters and live loss curves in the cloud.*

**Deliverables:**
- [ ] **A registered WandB account** and a local terminal authentication (`wandb login`).
- [ ] **A `config` dictionary** in your script containing all hyperparameters (learning rate, batch size, epochs, architecture name).
- [ ] **An integration in your Week 3 training loop** that initializes a wandb run using your config dictionary.
- [ ] **A cloud dashboard link** proving that your training loop successfully streamed Training Loss, Training Accuracy, Validation Loss, and Validation Accuracy to a live WandB web chart over 5 epochs.

### 🟨 THURSDAY: Advanced Diagnostics (Watching the Engine)
*Task: A model can train successfully while still suffering from vanishing gradients or dead neurons. You must configure your tracking tool to watch the internal state of the model, not just the output loss.*

**Deliverables:**
- [ ] **A WandB Model Watcher:** Configure WandB to "watch" your PyTorch model. Set the logging frequency so it captures data every X batches.
- [ ] **Histogram tracking:** Verify on your WandB web dashboard that you can see the distribution of your model's weights and gradients. (This proves your gradients are flowing properly through the layers).
- [ ] **Visual prediction logging:** Modify your validation loop so that at the end of every epoch, it uploads exactly one batch of *images* to WandB, overlaid with both their Ground Truth label and the Model's Predicted label. 

### 🟩 FRIDAY: Beyond Accuracy (Precision, Recall & F1)
*Task: "Accuracy" is a deceptive metric. If a dataset has 90 cats and 10 dogs, a model that only ever guesses "Cat" will be 90% accurate, but completely useless. You must calculate true classification performance.*

**Deliverables:**
- [ ] **An evaluation script** that runs your trained model over the entire validation set and collects all predicted indices and true indices into two flat lists/tensors.
- [ ] **Manual or `torchmetrics` calculation** of three critical metrics for each class: 
    - **Precision:** (When it predicts Dog, how often is it right?)
    - **Recall:** (Out of all actual Dogs, how many did it find?)
    - **F1-Score:** (The harmonic mean of Precision and Recall).
- [ ] **Console output** logging these three metrics. Prove your model is not just blindly guessing the majority class.

### 🟦 SATURDAY: The Confusion Matrix 
*Task: Visualizing errors tells you exactly what the network is failing to learn. You will map out the false positives and false negatives.*

**Deliverables:**
- [ ] **A Confusion Matrix calculation** generating an $N \times N$ matrix (where $N$ is your number of classes).
- [ ] **A visual heatmap plotting script** utilizing `matplotlib` and `seaborn`. The X-axis must be "Predicted Class" and the Y-axis must be "True Class".
- [ ] **A saved image artifact** (`confusion_matrix.png`). Ensure the plot includes annotated numbers in the grid squares and a color bar, so it is instantly readable.

### 🟪 SUNDAY: The Standalone Inference CLI
*Task: Your model is trained, logged, and evaluated. Now, separate the usage from the training pipeline. Write a script that a non-ML developer could use to predict on a single raw image.*

**Deliverables:**
- [ ] **An `inference.py` script** that takes command-line arguments (e.g., `python inference.py --image path/to/my_cat.jpg --weights checkpoints/best_model.pth`).
- [ ] **A raw-image ingestion pipeline** inside the script that opens a single PIL image, applies the exact same validation transforms (resizing, padding, normalization) used during training, and adds a batch dimension (`unsqueeze`).
- [ ] **A cleanly formatted console printout** displaying the predicted class string ("Dog" or "Cat") and the raw probability percentage of that prediction (using a Softmax function on the final logits).
- [ ] **A final Git push** containing your modular model class, your `inference.py` script, and your `confusion_matrix.png`.