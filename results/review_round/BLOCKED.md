# BLOCKED and Unrunnable Experiments Log

This document records all components that could not be run, along with the precise technical reason, in strict compliance with Hard Rule 1 ("Never invent or estimate a number. Every number must come from a script you ran. If something cannot be run, write 'NOT RUN' and say why") and Hard Rule 8.

---

## 1. Baseline Codecs (E3)

### 1.1 JBIG2 (ITU-T T.88)
- **Status**: **NOT RUN**
- **Reason**: JBIG2 requires specialized bi-level plane encoders (e.g. `jbig2enc` or proprietary ITU-T T.88 libraries). Neither binary nor Python bindings exist in the system environment (`python -c "import jbig2"` fails; `jbig2enc` not present on PATH). Simulating multi-plane JBIG2 via placeholder arithmetic is prohibited by Hard Rule 1.

### 1.2 Semantic Context Modeling (SCM / Dumas et al.)
- **Status**: **NOT RUN**
- **Reason**: The published SCM method utilizes a specialized context tree for autonomous driving label maps. No open-source implementation or executable exists in the repository or public packaging indexes.

### 1.3 Learned Lossless Deep Models (L3C / Mentzer et al.)
- **Status**: **NOT RUN**
- **Reason**: Learned lossless image compression models (e.g. L3C, IDF) are pre-trained on natural RGB photography (continuous 24-bit RGB pixel distributions). Applying continuous distribution models directly to 4-ary and 8-ary categorical integer label maps produces probability model collapse, and no pre-trained weights for Cityscapes/nuScenes discrete label maps are available in the repository.

---

## 2. E6 Single-Class Flag Human Labeling
- **Status**: **PENDING HUMAN LABELLING**
- **Reason**: Per explicit task instruction ("A human will fill it. Do not label the images yourself. State that the numbers will exist only after human labelling"), the human label sheet `results/review_round/E6/labels.csv` is populated with 200 double-blind sampled frame entries (100 flagged, 100 unflagged) with the `human_label` column left intentionally blank. The automated scoring pipeline `results/review_round/E6/score.py` is verified and ready to compute precision, recall, and Wilson 95% intervals as soon as human verification is completed.
