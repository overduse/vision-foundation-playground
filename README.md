# Vision Foundation Playground 🧪

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)]()

A modular codebase for exploring and experimenting with modern Computer Vision Foundation Models. This repository focuses on **Zero-shot** and **Few-shot** learning applications using state-of-the-art models.

The goal is to build a reusable library (`src`) to quickly prototype different downstream tasks (in `projects`), starting with CLIP and extending to SAM (Segment Anything Model).

## 🗺️ Roadmap

- [x] **Project Structure Setup**: Initialize modular architecture.
- [ ] **CLIP Integration**:
    - Wrap OpenAI/OpenCLIP models for easy inference.
    - Implement Zero-shot Image Classification.
- [ ] **SAM (Segment Anything)**:
    - Integrate SAM for prompt-able segmentation.
    - Implement interactive masking tools.
- [ ] **Advanced Pipelines**:
    - **CLIP + SAM**: Zero-shot semantic segmentation (detect objects with text, segment with SAM).
    - Few-shot adaptation experiments.

## 📂 Directory Structure

The repository is organized to separate core logic from experimental scripts:

```text
.
├── configs/          # Configuration files (.yaml) for models/experiments
├── data/             # Dataset storage (Ignored by Git)
├── notebooks/        # Jupyter notebooks for quick exploration & prototyping
├── projects/         # Standalone scripts for specific tasks (e.g., classifier)
├── src/              # Core library code
│   ├── models/       # Model wrappers (CLIP, SAM, etc.)
│   └── utils/        # Shared utilities (Image IO, visualization)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
