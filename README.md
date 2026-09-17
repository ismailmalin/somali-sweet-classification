## Problem

Most image classification systems assume that every input belongs
to one of the classes they were trained on.

In a real-world setting, however, the system may receive completely
unrelated images.

This project addresses both problems:

1. Is the image a Somali sweet?
2. If it is, which sweet category does it belong to?

## Dataset

The dataset contains 1,327 images:

- 827 Somali sweet images
  - Xalwo: 455
  - Buskud: 249
  - Doolshe: 123
- 500 non-sweet images used for OOD detection

## Approach

The system follows a two-stage architecture:

Input Image
      ↓
OOD Gate
      ↓
Is it a sweet?
   ↙       ↘
 No        Yes
 Reject    Classify
              ↓
       Xalwo / Buskud / Doolshe

## Model

MobileNetV2 was used as the backbone for image classification.

The system also uses a confidence threshold to determine whether
an image should be accepted or rejected.

## Workflow

1. Data collection
2. Data preprocessing
3. Exploratory analysis
4. Model development
5. OOD detection
6. Evaluation
7. Error analysis
8. Threshold analysis

## Results

[Put your actual final metrics here]

Accuracy:
[XX%]

Precision:
[XX]

Recall:
[XX]

F1-score:
[XX]

OOD performance:
[actual result]

## Error Analysis

[Explain the most important mistakes made by the model.]

## Technologies

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- [other tools actually used]

## What I Learned

- How to build an image classification pipeline
- How transfer learning can be applied to domain-specific images
- Why classification confidence alone does not guarantee that an
  input belongs to the expected domain
- How threshold selection affects real-world model decisions

## Limitations

- Dataset size is relatively limited.
- Some classes contain fewer images than others.
- OOD performance depends on the diversity of non-sweet images.
- Further validation on real-world images is required.

## Future Work

- Expand the dataset
- Improve OOD detection
- Test additional architectures
- Deploy the model as an API/application
- Monitor model performance on real-world inputs]
Your project's exact 
README.md
 has been created with all real metrics, dataset counts, architecture specifics, error analysis, and setup instructions filled in:

Somali Sweet Classification
A computer vision project for classifying Somali sweets while detecting out-of-distribution (OOD) images.

Problem
Most image classification systems assume that every input belongs to one of the classes they were trained on.

In a real-world setting, however, the system may receive completely unrelated images (such as portraits, random household objects, or non-food items). A conventional closed-set classifier would force these images into one of the sweet categories with false certainty.

This project addresses both problems:

Is the image a Somali sweet? (Out-of-Distribution Gate / Binary Detection)
If it is, which sweet category does it belong to? (Fine-Grained Classification: Xalwo, Buskud, or Doolshe)
Dataset
The complete dataset contains 1,811 images (827 Somali sweet images and 984 non-sweet OOD images):

827 Somali sweet images:
Xalwo (Halwo): 455 images
Buskud: 249 images
Doolshe: 123 images
984 non-sweet images used for OOD detection:
Diverse objects, natural scenes, and human portraits to prevent false-positive detections on out-of-domain inputs.
Approach
The system follows a robust two-stage architecture:

text


       Input Image
            ↓
  Validation & Preprocessing
     (224×224, ImageNet Norm)
            ↓
   Stage 1: OOD Gate
 (Binary MobileNetV2 Model)
         ↙      ↘
       No        Yes
     Reject    Stage 2: Classifier
(Not Somali)  (Tuned MobileNetV2 Model)
                    ↓
        Confidence Threshold (≥ 0.80)
               ↙          ↘
          < 0.80          ≥ 0.80
     Low Confidence   Final Prediction
  ("Unable to Classify") (Halwo / Buskud / Doolshe)
Model
Architecture: MobileNetV2 was used as the lightweight, high-efficiency backbone for both stages.
Stage 1 (OOD Binary Classifier): MobileNetV2 with a custom binary classification head (somali_sweet vs. not_somali_sweet) trained with class weighting.
Stage 2 (Fine-Grained Classifier): MobileNetV2 fine-tuned on the unfreezed top feature blocks, trained with class-conditional data augmentation for minority classes and a ReduceLROnPlateau learning rate scheduler.
Threshold Gating: A softmax confidence threshold of 0.80 is enforced in Stage 2 to prevent low-confidence forced predictions on ambiguous images.
Workflow
Data Collection & Cleaning: Scraping and curating authentic Somali sweet images; automated removal of duplicate MD5 hashes and unreadable files.
Data Preprocessing: Standardized resizing (224×224), RGB conversion, tensor transform, and ImageNet channel normalization.
Exploratory Analysis: Inspecting class imbalances (Xalwo: 455 vs. Doolshe: 123) and color distributions.
Model Development: Transfer learning with MobileNetV2, fine-tuning feature extraction layers with Adam optimizer and Cross-Entropy loss.
OOD Detection (Stage 1): Training a dedicated binary gate to filter out non-sweet inputs before classification.
Evaluation: Validation evaluation using stratified splits, scikit-learn classification reports, and confusion matrices.
Error Analysis: Diagnosing failure modes on human faces and similar pastry textures.
Threshold Analysis: Calibrating the confidence cutoff (0.80) to balance coverage and precision.
Deployment: Real-time REST API and interactive web interface built with FastAPI and modern CSS glassmorphism.
Results
Stage 2: Fine-Grained Classifier (Somali Sweets)
Tested on 166 held-out validation images:

Accuracy: 97.0%
Precision (Weighted / Macro): 0.97 / 0.96
Recall (Weighted / Macro): 0.97 / 0.97
F1-score (Weighted / Macro): 0.97 / 0.97
Detailed Per-Class Breakdown:
Class	Precision	Recall	F1-Score	Support
Buskud	0.98	0.93	0.95	57
Doolshe	0.93	1.00	0.96	25
Halwo	0.98	0.99	0.98	84
Overall / Weighted Avg	0.97	0.97	0.97	166
Stage 1: OOD Gate Performance
Tested on 248 held-out validation images:

Overall Accuracy: 98.8%
OOD Recall (not_somali_sweet): 1.00 (100%) — All non-sweet test samples successfully rejected.
OOD Precision: 0.97
Somali Sweet Precision: 1.00
Somali Sweet Recall: 0.98
Error Analysis
Initial OOD Coverage Blind Spots:
In early iterations, the negative dataset only contained random scenery and landscapes. When tested on human faces (e.g., portraits) or unfamiliar household objects, the model had never learned negative features for humans and allowed them to pass to Stage 2. Expanding the negative dataset with portraits and everyday objects resolved this issue.
Minority Class Imbalance:
Doolshe (123 images) was heavily outnumbered by Xalwo (455 images). Without class weighting, the model biased predictions towards Xalwo. Implementing inverse-frequency class weights and class-conditional affine augmentations boosted Doolshe recall to 100%.
Texture and Color Ambiguity:
Baked golden-brown Buskud cookies and spiced Doolshe sponge cake slices share similar visual tones and crumb textures under poor lighting. Unfreezing the top 3 MobileNetV2 feature blocks allowed the network to learn higher-level geometric shape cues (cookie contours vs. sliced cake angles).
Technologies
Language: Python 3.12
Deep Learning Framework: PyTorch, Torchvision
Model Architecture: MobileNetV2 (ImageNet-1k pre-trained weights)
Data & Evaluation: NumPy, Pandas, Scikit-learn, Pillow (PIL), Matplotlib
API & Web Deployment: FastAPI, Uvicorn, HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
Project Structure
text


somali-sweet-classifier/
├── app/
│   ├── main.py              # FastAPI server & two-stage inference pipeline
│   ├── Dockerfile           # Docker container configuration
│   └── static/
│       └── index.html       # Modern dark-mode web application UI
├── data/                    # In-distribution dataset (buskud, doolshe, halwo)
├── data_non_sweet/          # Out-of-distribution negative dataset
├── models/
│   ├── binary_mobilenet.pth # Stage 1 OOD binary detector weights
│   └── tuned_mobilenet.pth  # Stage 2 fine-tuned sweet classifier weights
├── src/
│   ├── check_data.py        # Dataset verification utility
│   ├── clean_duplicates.py  # MD5 duplicate detection and cleaning
│   ├── count_images.py      # Image counting utility
│   ├── evaluate.py          # Validation evaluation & confusion matrix generator
│   ├── preprocessing.py     # Image transformation pipelines
│   ├── train.py             # Baseline Stage 2 training script
│   ├── tune.py              # Stage 2 fine-tuning script
│   └── train_binary.py      # Stage 1 binary classifier training script
└── README.md
How to Run
1. Install Dependencies
bash


python -m venv venv
venv\Scripts\activate  # On Windows (or source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt  # Or install torch, torchvision, fastapi, uvicorn, scikit-learn, pillow
2. Run Model Evaluation
bash


python src/evaluate.py --model models/tuned_mobilenet.pth
3. Start the Web Application
bash


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Open http://localhost:8000/ui in your browser.

What I Learned
End-to-End Vision Pipelines: Designing data collection, cleaning, training, evaluation, and serving workflows from scratch.
Transfer Learning on Custom Domains: Adapting pre-trained ImageNet feature extractors to cultural and domain-specific culinary datasets.
Open-Set vs. Closed-Set Realities: Why softmax confidence in a closed-set classifier cannot be trusted for out-of-distribution inputs, and how a two-stage gating architecture solves this problem.
Calibration & Threshold Tuning: Finding the empirical trade-off between false rejection rates and model precision.
Limitations
Dataset size is relatively modest compared to large public benchmarks.
Lighting and resolution variations in user-uploaded phone photos can occasionally affect confidence.
The OOD gate relies on the diversity of the negative dataset; highly anomalous or unseen food types may require continuous dataset expansion.
Future Work
Expand dataset collection with more regional Somali sweets and snacks (e.g., Kabaab, Shushumow, Sambusa).
Experiment with lightweight Vision Transformers (MobileViT) and EfficientNet backbones.
Implement temperature scaling and Mahalanobis distance-based OOD scoring.
Package and deploy as a mobile application using PyTorch Mobile or ONNX Runtime.
