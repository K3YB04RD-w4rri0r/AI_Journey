[[Computer Vision Roadmap - 1]]

- **Project 1 (Weeks 10-21):** A YOLO Object Detection model, optimized via ONNX to TensorRT, focused on maximizing FPS (Frames Per Second).
- **Project 2 (Weeks 15-21):** A Vision Transformer (ViT) Image Classifier, deployed as a FastAPI microservice in a Docker container, focusing on precision/recall in a specialized domain.

To make an "uncompromising" portfolio that will actually get an ML Engineering Lead to reply to your Week 32 cold emails, **you must avoid toy datasets** (No dogs vs. cats, no MNIST, no basic COCO). You need messy, real-world problems.

---

### 🔥 PROJECT 1: The TensorRT / Edge Deployment Project

**The Stack:** YOLOv10, PyTorch, ONNX, TensorRT, Video Inference. **The Goal:** Prove you can take a heavy model and make it run blisteringly fast for real-time applications.

### Option A: Traffic/Dashcam Analytics (Emergency Vehicle & Threat Detection)

Instead of just tracking cars (boring), fine-tune YOLO to specifically detect emergency vehicles (ambulances, firetrucks), jaywalking pedestrians, or aggressive lane-changes.

- **Dataset:** BDD100K (Berkeley DeepDrive) or DAWN dataset.
- **Pros:** Video inference makes for a highly engaging 15-second YouTube demo. TensorRT FPS gains are incredibly obvious on video.
- **Cons:** High compute required to process video frames. Bounding boxes for distant objects can be frustratingly small.
- **The Flex Factor:** Proves you can handle temporal data (video streams) and optimize for real-time autonomous driving/smart city use cases.

### Option B: Automated Retail / Dense Shelf Inventory

Fine-tune YOLO to detect specific consumer goods on wildly crowded supermarket shelves, identifying out-of-stock gaps or misplaced items.

- **Dataset:** SKU-110K dataset (features densely packed items).
- **Pros:** Massive commercial value. Every retail tech company (Amazon Go, Trax, etc.) solves this exact problem.
- **Cons:** "Dense Object Detection" is notorious for crashing standard NMS (Non-Max Suppression). You will have to heavily tune your IoU thresholds.
- **The Flex Factor:** Proves you can handle overlapping bounding boxes and extreme clutter, which breaks out-of-the-box YOLO models.

### Option C: Industrial Manufacturing Defect Detection

Detecting scratches, dents, or missing components on a fast-moving assembly line (e.g., PCBs or metal welding).

- **Dataset:** NEU Surface Defect Database or PKU-Market-PCB.
- **Pros:** Manufacturing MLOps is booming. The definition of a "true positive" is very strict, which forces you to deeply understand your loss function (Week 11/12).
- **Cons:** Datasets are often small, requiring heavy data augmentation (Week 7 `albumentations` skills will be tested).
- **The Flex Factor:** You can pitch the TensorRT optimization as a business cost-saver: _"Optimized inference from 22 FPS to 85 FPS, allowing the assembly line conveyor belt to run 3x faster."_

---

### 🧬 PROJECT 2: The ViT / API Microservice Project

**The Stack:** ViT (Hugging Face), `einops`, FastAPI, Docker. **The Goal:** Prove you understand state-of-the-art attention mechanisms and can wrap them in a production-ready, containerized backend.

### Option A: Histopathology (Cancer Tissue Classification)

Classifying whether patches of gigapixel microscope slide images contain malignant or benign tissue. (Directly aligns with the "medical data" mentioned in Week 18).

- **Dataset:** PatchCamelyon (PCam) or RSNA Breast Cancer.
- **Pros:** ViTs excel at patch-based processing, making this the perfect architectural fit. Extremely high impact.
- **Cons:** Medical data is highly imbalanced (Week 8 skills required). You will have to aggressively optimize for False Negatives (Recall), because missing cancer is worse than a false alarm.
- **The Flex Factor:** Proves you understand the statistical gravity of ML (Precision vs Recall curves, F1 scores) rather than just looking at raw "Accuracy."

### Option B: Satellite Imagery Disaster Assessment

Using high-resolution satellite imagery to classify areas as "destroyed," "flooded," or "safe" post-natural disaster.

- **Dataset:** xBD Dataset (xView2) or EuroSAT.
- **Pros:** Visually stunning for the README. High global relevance (climate tech, defense, insurance).
- **Cons:** Satellite images are often `.TIFF` files with multiple channels (not just RGB). You will have to write a very custom `torch.utils.data.Dataset` (Week 2) to handle this.
- **The Flex Factor:** By successfully writing a custom data loader for non-standard, multi-spectral image formats, you prove you aren't reliant on basic `torchvision.datasets`.

### Option C: Crop Disease / Precision Agriculture

Classifying specific diseases on plant leaves taken in messy, real-world field conditions (variable lighting, shadows, dirt).

- **Dataset:** Plant Pathology (Kaggle FGVC) — _Do not use "Plant Village", it is too clean with white backgrounds._
- **Pros:** Clear, interpretable features. Easy to build a dummy front-end later if you want to show it as a "farmer's mobile app."
- **Cons:** The model will try to cheat by looking at the background dirt rather than the leaf (spurious correlations).
- **The Flex Factor:** You can use Week 28 (DINOv2) to extract features and visualize _where_ the ViT is looking, proving you know how to debug model explainability.

---

### 🧠 How to Make Your Final Choice

1. **Pick one from Project 1 and one from Project 2 right now.** Don't wait until Week 10. Write them into your schedule.
2. **Ensure they are visually distinct.** Don't do dashcams for Project 1 and satellite cars for Project 2. Show range. (E.g., Manufacturing Defects for Project 1 + Medical Cancer for Project 2).
3. **The Golden Rule of the 15-Second Demo (Weeks 30-32):** Hiring managers do not care about your training loop. They care about your deployment. Your final GitHub READMEs must lead with a GIF showing the model running, the REST API returning a JSON payload, or a graph showing the FPS jump from PyTorch to TensorRT.

**Recommendation:**

- **Project 1:** Traffic/Dashcam Analytics. (It provides the most visceral, easily understood proof of TensorRT FPS acceleration).
- **Project 2:** Histopathology/Cancer Detection. (It perfectly suits the ViT architecture and forces you to grapple with severe class imbalance).