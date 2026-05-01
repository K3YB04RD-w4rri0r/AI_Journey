[[Survival Guide]]

_Budget: 10–14 hours per week. If you miss a day, catch up on the weekend. Use the Sunday Discord sync._

### **Phase 1: PyTorch Fluency (Weeks 1-5)** 
[[Week 1]], [[Week 2]]

- **Week 1:** Install Conda, VS Code, PyTorch. Write a script to multiply and reshape tensors (`.view()`, `.squeeze()`).
- **Week 2:** Build a custom `torch.utils.data.Dataset` using a messy image dataset. Handle resizing and normalization manually.
- **Week 3:** Write a raw training loop (Forward, Loss, `.backward()`, `.step()`, `.zero_grad()`). Overfit on a single batch.
- **Week 4:** Build a basic CNN. Integrate `wandb` for logging. Plot a **Confusion Matrix**.
- **Week 5: BUFFER WEEK.**

### **Phase 2: Math to Code - ResNet (Weeks 6-9)**

- **Week 6 (Read & Code):** Read the _ResNet_ paper (3 hours). Re-implement the Residual Block from scratch (no `torchvision`) and assemble a mini-ResNet (7-10 hours).
- **Week 7:** Load a pre-trained ResNet-50. Freeze layers. Add heavy `albumentations` (Cutout, ColorJitter).
- **Week 8:** Train on an imbalanced dataset. Calculate Precision, Recall, and F1-Score. Balance classes via weighted loss or oversampling.
- **Week 9: BUFFER WEEK.**

### **Phase 3: The Object Detection Crucible (Weeks 10-14)**

- **Week 10 (Read & Code):** Read the _YOLOv1_ paper (3 hours). Code Intersection over Union (IoU) and Non-Max Suppression (NMS) from scratch in PyTorch (10 hours).
- **Week 11:** **YOLO Loss Part 1:** Focus entirely on grid cell assignment, masking, and the coordinate loss λ-coord.
- **Week 12:** **YOLO Loss Part 2:** Implement the objectness and class probability loss. Combine it and debug NaN errors.
- **Week 13:** Use Ultralytics (YOLOv10). Fine-tune on a custom video dataset (e.g., dashcam footage) and run inference.
- **Week 14: BUFFER WEEK.**

### **Phase 4: MLOps & The TensorRT Mountain (Weeks 20-26)**

- **Week 20:** Build a FastAPI backend that accepts an image via POST request and returns a JSON bounding box prediction.
- **Week 16:** Write a `Dockerfile` for your API. Run the container locally.
- **Week 17:** Export your YOLO model to **ONNX**. Run ONNX Runtime inference using Python.
- **Week 18:** **TensorRT Part 1 (The Environment):** Install NVIDIA TensorRT, CUDA, and cuDNN. Your goal this week is purely to get the environment working without dependency errors.
- **Week 19:** **TensorRT Part 2 (The Export):** Convert your ONNX model to a TensorRT `.engine` file.
- **Week 20:** **TensorRT Part 3 (The Inference):** Write the inference script. Measure the FPS difference between PyTorch, ONNX, and TensorRT.
- **Week 21: BUFFER WEEK.** (You will absolutely need this to recover from Week 23-25).

### **Phase 5: The Attention Era (Weeks 20-26)**

- **Week 22 (Read & Code):** Read the _ViT_ paper. Learn `einops`. Code the ViT Patch Embedding layer (converting an image into a sequence of tokens).
- **Week 23:** **Attention Part 1:** Code the scaled dot-product attention and the Q, K, V linear projections.
- **Week 24:** **Attention Part 2:** Implement the Multi-Head split using `einops`, the MLP block, and assemble the full Transformer Encoder block.
- **Week 25:** Learn the Hugging Face `transformers` and `datasets` ecosystem. Fine-tune a ViT on medical data.
- **Week 26: BUFFER WEEK.**

### **Phase 6: Kaggle Wisdom & SSL (Weeks 27-29)**

- **Week 27:** Go to a finished Kaggle CV competition (e.g., RSNA Mammography). Read the top 5 Solution Writeups. Document 3 engineering tricks they used to win.
- **Week 28 (Read & Code):** Read about Meta's DINOv2. Load a pre-trained DINOv2 model, extract features from an unlabelled dataset, and run K-Means clustering to see it group images automatically.
- **Week 29: BUFFER WEEK.**

### **Phase 7: The Portfolio & Outreach (Weeks 30-32)**

- **Week 30:** Polish Project 1 (TensorRT). Write the README detailing FPS gains. Record a 15-second screen capture. Upload as Unlisted to YouTube.
- **Week 31:** Polish Project 2 (FastAPI/ViT). Write the README. Record a 15-second demo. Upload to YouTube.
- **Week 32:** Profile update & Cold Outreach. Send the "15s demo link" template to 30 ML Engineers/Leads.