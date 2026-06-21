# AERUX: AI-Powered Multimodal Intracranial Aneurysm Detection Platform
AERUX is a full-stack AI-powered medical imaging platform for candidate-level intracranial aneurysm detection and anatomical localization across CTA, MRA, and MRI scans. The system combines a modern Next.js frontend, FastAPI backend, multimodal deep learning pipeline, and Grok-powered chatbot integration to provide an interactive research prototype for computer-aided aneurysm analysis.

## Project Overview:
Intracranial aneurysms are serious cerebrovascular abnormalities that may rupture and lead to subarachnoid hemorrhage. Early detection is clinically important, but manual interpretation of neuroimaging scans can be time-consuming, operator-dependent, and challenging across different imaging modalities.

This final year project presents an end-to-end AI-based platform that detects aneurysm candidates and predicts their anatomical location using multimodal medical imaging data. The deep learning pipeline uses 2.5D representations from CTA, MRA, and MRI scans, modality-specific CNN encoders, and an attention-based fusion mechanism to combine diagnostic information across modalities.
The project also includes a complete web application, allowing users to interact with the system through a frontend interface, backend APIs, and an AI chatbot assistant.

## Key Features:
* Multimodal aneurysm analysis using CTA, MRA, and MRI scans
* Candidate-level binary aneurysm detection
* Multi-label anatomical localization across 13 arterial regions
* 2.5D slice-based representation using seven consecutive axial slices
* Attention-based late fusion for modality-adaptive feature weighting
* ResNet50 and DenseNet121 backbone comparison
* FastAPI backend for API-based model interaction
* Next.js, React, and TypeScript frontend
* Grok primary and fallback chatbot models for deep reasoning support
* API-based frontend-backend integration
* Research-oriented architecture for medical image analysis and clinical AI prototyping

## System Architecture:
AERUX consists of three main components:

### 1. Frontend:
The frontend is built using:
* Next.js
* React
* TypeScript
* API-based integrations
* Next.js App Router
* Optimized font loading using `next/font`

The frontend provides the user-facing interface for interacting with the aneurysm detection platform, communicating with backend APIs, and accessing chatbot-based assistance.

### 2. Backend:
The backend is built using:
* FastAPI
* Python
* REST API endpoints
* Model inference integration
* Chatbot API integration
* Frontend-backend communication layer

The backend handles requests from the frontend, connects the application to the AI model pipeline, and manages communication with the Grok chatbot system.

### 3. AI Model Pipeline:
The AI pipeline is designed for multimodal aneurysm detection and localization. CTA, MRA, and MRI inputs are converted into 2.5D representations and processed through modality-specific encoders. The extracted features are fused using an attention-based projection mechanism and passed into task-specific prediction heads.

The model performs:
* Binary aneurysm classification
* Multi-label localization across 13 anatomical arterial sites
* Auxiliary segmentation-style regularization during experimentation

## Research Methodology:
The model pipeline includes the following stages:

1. DICOM loading and volume reconstruction
2. 2.5D slice extraction using seven axial slices
3. Modality-specific preprocessing
4. CTA intensity normalization using HU windowing
5. MRA/MRI correction using N4 bias field correction
6. Vessel enhancement using a Sato vesselness filter
7. Feature extraction using ResNet50 and DenseNet121 encoders
8. Attention-based feature fusion
9. Detection and localization through task-specific heads

The framework was evaluated using the RSNA Intracranial Aneurysm Detection Challenge dataset, which contains CTA, MRA, and MRI imaging series with binary aneurysm labels and multi-label anatomical localization annotations.

## Dataset:
The project uses the RSNA Intracranial Aneurysm Detection Challenge dataset.

Dataset details:
* Total imaging series: 4,348
* Modalities: CTA, MRA, MRI
* Training samples: 3,478
* Held-out test samples: 870
* Binary labels: Aneurysm / No Aneurysm
* Localization labels: 13 anatomical arterial locations

The evaluation was performed on pre-cropped candidate-level samples. This means the reported results assume that a candidate proposal mechanism has already identified the region of interest.

## Anatomical Localization Targets:
The system supports localization across 13 intracranial arterial regions:
* Left infraclinoid internal carotid artery
* Right infraclinoid internal carotid artery
* Left supraclinoid internal carotid artery
* Right supraclinoid internal carotid artery
* Left middle cerebral artery
* Right middle cerebral artery
* Anterior communicating artery
* Left anterior cerebral artery
* Right anterior cerebral artery
* Left posterior communicating artery
* Right posterior communicating artery
* Basilar tip
* Other posterior circulation

## Model Performance:
The final multimodal fusion models were evaluated on a held-out test set of 870 pre-cropped candidate samples.

| Metric               | ResNet50 Fusion | DenseNet121 Fusion |
| -------------------- | --------------: | -----------------: |
| Accuracy             |          97.59% |             98.39% |
| Detection AUC-ROC    |           0.997 |              0.996 |
| Sensitivity / Recall |           0.992 |              0.976 |
| Specificity          |           0.964 |              0.990 |
| Precision            |           0.954 |              0.986 |
| F1-Score             |           0.972 |              0.981 |
| Weighted AUC         |           0.913 |              0.911 |
| Correct / Incorrect  |        849 / 21 |           856 / 14 |

## Backbone Comparison:
Both ResNet50 and DenseNet121 achieved strong performance in the multimodal fusion setting.
The ResNet50 fusion model achieved slightly higher Detection AUC-ROC, sensitivity, and weighted AUC. This makes ResNet50 particularly valuable in medical screening contexts where minimizing false negatives is important.
The DenseNet121 fusion model achieved higher overall accuracy, precision, specificity, F1-score, and fewer total misclassifications. This shows that DenseNet121 produced stronger overall classification balance and fewer false positives on the held-out test set.
Overall, ResNet50 was slightly stronger for sensitivity-focused detection, while DenseNet121 showed stronger balanced classification performance.

## Modality-Specific Results:
Before fusion, modality-specific models were trained separately for CTA, MRA, and MRI.

### ResNet50 Modality-Specific Performance:
| Modality | Samples | Detection AUC | Accuracy | Precision | Recall | Weighted AUC |
| -------- | ------: | ------------: | -------: | --------: | -----: | -----------: |
| CTA      |     359 |        0.9999 |   99.72% |     0.995 |  1.000 |        0.917 |
| MRA      |     245 |        0.9971 |   98.78% |     1.000 |  0.975 |        0.929 |
| MRI      |     266 |        1.0000 |  100.00% |     1.000 |  1.000 |        0.927 |

### DenseNet121 Modality-Specific Performance:
| Modality | Samples | Detection AUC | Accuracy | Precision | Recall | Weighted AUC |
| -------- | ------: | ------------: | -------: | --------: | -----: | -----------: |
| CTA      |     359 |        0.9939 |   98.61% |     0.979 |  0.995 |        0.878 |
| MRA      |     245 |        0.9819 |   98.78% |     1.000 |  0.975 |        0.925 |
| MRI      |     266 |        1.0000 |  100.00% |     1.000 |  1.000 |        0.934 |

These results show that both backbone architectures achieved strong modality-specific performance. ResNet50 performed slightly better for CTA and MRA detection AUC, while DenseNet121 achieved the highest weighted AUC for MRI.

## Chatbot Integration:
AERUX includes an AI chatbot assistant integrated through Grok.
The chatbot uses:
* Grok primary model
* Grok fallback model for deep reasoning
* API-based communication with the backend

The chatbot is designed to help users interact with the system more naturally, understand platform outputs, and receive reasoning-style responses related to the workflow.

## Tech Stack:
### Frontend:
* Next.js
* React
* TypeScript
* Tailwind CSS
* API-based integration
* Next.js App Router

### Backend:
* FastAPI
* Python
* REST APIs
* Model inference endpoints
* Grok API integration

### AI / Machine Learning:
* PyTorch
* ResNet50
* DenseNet121
* Attention-based fusion
* Multi-task learning
* Medical image preprocessing
* DICOM processing
* 2.5D image representation
* Candidate-level classification
* Multi-label localization

### Dataset:
* RSNA Intracranial Aneurysm Detection Challenge dataset
* CTA, MRA, and MRI imaging series
* Binary aneurysm labels
* Multi-label anatomical localization labels

## Getting Started:
This project contains a Next.js frontend and a FastAPI backend.

### Prerequisites:
Make sure you have the following installed:
* Node.js
* npm, yarn, pnpm, or bun
* Python 3.10+
* pip
* Git

## Frontend Setup:
Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

You can also use:

```bash
yarn dev
# or
pnpm dev
# or
bun dev
```

Open the application in your browser:

```bash
http://localhost:3000
```

You can start editing the frontend by modifying:

```bash
app/page.tsx
```

The page auto-updates as you edit the file.

## Backend Setup:
Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will usually run at:

```bash
http://localhost:8000
```

API documentation can usually be accessed at:

```bash
http://localhost:8000/docs
```

## Environment Variables:
Create a `.env` file for API keys and configuration values.

Example:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
GROK_API_KEY=your_grok_api_key_here
MODEL_PATH=path_to_model_weights
```

Update the variable names based on your actual implementation.

## Project Structure:
```bash
AERUX/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── models/
│   ├── services/
│   ├── requirements.txt
│   └── README.md
│
├── model/
│   ├── preprocessing/
│   ├── inference/
│   └── weights/
│
└── README.md
```

## Research Contribution:
The main research contribution of AERUX is the development of a multimodal attention-based fusion framework for intracranial aneurysm candidate classification and anatomical localization.
Unlike single-modality systems, AERUX integrates CTA, MRA, and MRI information into a unified architecture. The system uses modality-specific encoders and attention-based feature transformation to combine complementary imaging information across modalities.
The project also compares ResNet50 and DenseNet121 backbones, showing that both architectures perform strongly, with ResNet50 being slightly stronger for sensitivity-focused detection and DenseNet121 producing stronger balanced classification metrics.

## Important Disclaimer:
This project is a research prototype developed for academic purposes. It is not a certified medical device and should not be used for real clinical diagnosis, treatment planning, or emergency medical decision-making.

The reported results are based on candidate-level evaluation using pre-cropped samples and assume the presence of a reliable candidate proposal mechanism. Full-volume detection, external validation, clinical testing, and regulatory approval would be required before any real-world medical use.

## Future Improvements:
* Validate the system on external multi-center datasets
* Add stronger model interpretability using Grad-CAM or saliency maps
* Validate segmentation outputs using ground-truth masks
* Expand chatbot capabilities for detailed report interpretation
* Add authentication and role-based access
* Deploy the frontend and backend to production cloud services
* Improve model monitoring and inference logging

## Deployment:
The frontend can be deployed using Vercel, while the backend can be deployed using platforms such as Render, Railway, AWS, Azure, or Google Cloud.
For frontend deployment, the easiest option is the Vercel platform from the creators of Next.js.

## Learn More:
Useful resources:
* Next.js Documentation
* React Documentation
* FastAPI Documentation
* PyTorch Documentation
* RSNA Intracranial Aneurysm Detection Challenge

## Authors:
Final Year Project developed by the AERUX team.

## License:
This project is intended for academic and research purposes.
