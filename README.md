# [NeurIPS 2025] SAM2Flow: Interactive Optical Flow Estimation with Dual Memory for in vivo Microcirculation Analysis
## Updates
**2025-12-02:** Training code & weights are coming soon!
## Abstract
Analysis of noninvasive microvascular blood flow can improve the diagnosis, prognosis, and management of many medical conditions, including cardiovascular, peripheral vascular, and sickle cell disease. This paper introduces SAM2Flow, an interactive optical flow estimation model to analyze long Oblique Back-illumination Microscopy (OBM) videos of in vivo microvascular flow. Inspired by the Segment Anything Model (SAM2), SAM2Flow enables users to specify regions of interest through user prompts for focused flow estimation. SAM2Flow also incorporates a dual memory attention mechanism, comprising both motion and context memory, to achieve efficient and stable flow estimations over extended video sequences. According to our experiments, SAM2Flow achieves SOTA accuracy in flow estimation with a fast inference speed of over $200$ fps on $512\times 512$ inputs. Based on the temporally robust flow estimation, SAM2Flow demonstrated superior performance in downstream physiological applications compared to existing models. 

## Highlights
- **SAM2 ViT context encoder** with lightweight adapters for biomedical inputs.
- **Prompt encoder + ROI decoder** so users can specify sparse point prompts per vessel/branch.
- **ROI-guided correlation lookup** to restrict matching to foreground vessels.
- **Dual memory bank** (motion + context) with attention-based retrieval for long OBM videos.
- Configurable via `configs/sam2flow_base.py` and instantiated though `core.Networks.build_network`.

## Quick Start
1. Ensure required Python dependencies (PyTorch, torchvision, yacs, numpy, scipy, scikit-image, etc.) are installed.
2. (Optional) Place pretrained SAM2 checkpoints under `core/Networks/SAM2` and update `encoder.pretrained_fondation` in the config.
3. Run a demo forward pass on random data:
   ```bash
   cd SAM2Flow
   python scripts/demo_sam2flow.py --frames 2
   ```
   The script loads the default config, builds SAM2Flow, and reports the output tensor shapes.

## Training & Datasets
- The dataset loader is implemented in `core/datasets/datasets_SAM.py` with prompt sampling utilities.
- To train, adapt the notebook pipelines (`notebooks/0_1_train_SEARAFT.ipynb`) or build a custom trainer leveraging `core.Networks.build_network(get_cfg())` and the provided dataloaders.
- Memory length, decoder depth, optimizer, and LR schedule are configurable under `configs/sam2flow_base.py` and `_CN.trainer`.

## Repository Layout
```
core/Networks/           # Model definitions (SAM2 encoder, SAM2Flow, update blocks)
core/datasets/           # Optical flow datasets + prompt sampling
core/utils/              # Augmentations, flow utilities
configs/                 # YACS configs (SAM2Flow defaults)
scripts/demo_sam2flow.py # Minimal demo entry point
notebooks/               # Data prep / legacy experiments
```
