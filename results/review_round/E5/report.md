# E5: Class-Wise and Spatial Staleness Benchmark Report

## 1. Executive Summary

This report evaluates class-wise and spatial staleness (Age of Information, AoI) across semantic categories and camera orientations for the $K=8$ class mapping on the complete nuScenes sweep dataset (10 scenes $\times$ 6 cameras = 60 streams $\times$ 100 frames = 6,000 frames evaluated over 5 random seeds, $N=30,000$ frame simulations per channel condition).

### Core Questions Addressed
1. **Does slicing ($S=1, 4, 8$) help small objects (Pedestrians, Vehicles) differently from large background regions (Road)?**
   - **Finding**: **Yes, with distinct operational trade-offs.**
     - **Spatial Concentration**: Pedestrians occupy only 1.33% of image pixels and are heavily concentrated in Slice 2 (rows 226–338, 85.1% of pedestrian pixels) and Slice 3 (rows 338–450, 14.2%). Road occupies 24.33% and is concentrated in Slice 3 (74.8%) and Slice 2 (24.8%).
     - **Resilience**: With whole-frame transmission ($S=1$), losing a packet drops all classes simultaneously. With horizontal slicing ($S=4, 8$), when a burst or Bernoulli drop corrupts an upper slice (e.g. Slice 0 or 1, which contain sky and distant structures), road and pedestrians are completely unharmed.
     - **Staleness Bound**: Because pedestrians are confined to very few slices, when the pedestrian-bearing slice *is* dropped, the pedestrian region remains stale even if the road is refreshed. Under burst channels ($T_B = 100$ ms), $S=8$ reduces pedestrian mean AoI from 120.3 ms to 119.5 ms, and vehicle mean AoI from 120.2 ms to 117.5 ms.
2. **Does horizontal slicing help lateral regions differently?**
   - **Lateral vs Frontal Cameras**: Lateral cameras (`CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`) show virtually identical AoI statistics to frontal cameras (`CAM_FRONT`, `CAM_BACK`) under independent channel fading (Mean AoI: 114.6 ms vs 114.5 ms at 10 Hz Bernoulli).
   - **Lateral Columns (within-frame left/right edges)**: Because horizontal slicing partitions the image vertically into horizontal bands, every horizontal slice spans the entire image width ($W=800$ px). Consequently, horizontal slicing offers identical delivery probability to lateral columns ($c < 200$ and $c \ge 600$) as it does to center columns ($200 \le c < 600$).

---

## 2. Experimental Setup

- **Dataset**: All 60 sweep streams from nuScenes (10 scenes $\times$ 6 cameras $\times$ 100 frames = 6,000 frames).
- **Resolution & Mapping**: Half resolution ($450 \times 800$), $K=8$ class mapping.
  - Class 1: Road (24.33% of pixels)
  - Class 6: Vehicle (4.10% of pixels)
  - Class 8: Pedestrian / Person / Rider (1.33% of pixels)
  - Static structures (Class 3), Vegetation (Class 4), Sky (Class 5), Two-wheeler (Class 7), Sidewalk (Class 2).
- **Channels Evaluated**:
  1. Bernoulli Packet Loss: $P_L = 10\%$.
  2. Gilbert-Elliott Burst Loss: Mean burst duration $T_B = 50$ ms, $P_L = 10\%$.
  3. Gilbert-Elliott Burst Loss: Mean burst duration $T_B = 100$ ms, $P_L = 10\%$.
- **Slicing Configurations**: $S \in \{1, 4, 8\}$ horizontal slices per frame.
- **Timing Parameters**:
  - $f = 10.0$ Hz ($\Delta t = 100.0$ ms)
  - $T_{\text{proc}} = 103.4$ ms
  - Packet transmission window: 5.0 ms for $S > 1$.
- **Statistical Rigor**:
  - 5 independent random seeds (seeds 0–4).
  - 10,000 scene-cluster bootstrap resamples over the 10 nuScenes scenes for 95% confidence intervals.

---

## 3. Results Summary Table (10 Hz, $T_{\text{proc}} = 103.4$ ms)

| Channel | Category | S=1 Mean AoI [95% CI] (ms) | S=4 Mean AoI [95% CI] (ms) | S=8 Mean AoI [95% CI] (ms) | S=1 p95 | S=4 p95 | S=8 p95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bernoulli (10%)** | **Pedestrian** | 115.3 [114.7, 115.9] | 114.6 [114.1, 115.1] | 114.7 [114.2, 115.2] | 203.4 | 203.4 | 203.4 |
| | **Vehicle** | 114.8 [114.3, 115.2] | 114.5 [114.0, 114.9] | 115.0 [114.5, 115.5] | 203.4 | 203.4 | 203.4 |
| | **Road** | 114.7 [114.3, 115.1] | 114.4 [114.0, 114.8] | 114.7 [114.3, 115.1] | 203.4 | 203.4 | 203.4 |
| | **Full Frame** | 114.7 [114.3, 115.2] | 114.6 [114.1, 115.0] | 114.6 [114.2, 115.0] | 203.4 | 203.4 | 203.4 |
| | Lateral Cameras | 114.9 [114.3, 115.4] | 114.6 [114.1, 115.1] | 114.8 [114.2, 115.3] | 203.4 | 203.4 | 203.4 |
| | Frontal Cameras | 114.5 [114.0, 115.1] | 114.5 [114.0, 114.9] | 114.3 [113.8, 114.8] | 203.4 | 203.4 | 203.4 |
| **Burst $T_B=50$ ms** | **Pedestrian** | 117.2 [115.4, 118.8] | 116.7 [115.2, 118.1] | 116.4 [114.8, 117.9] | 203.4 | 203.4 | 203.4 |
| | **Vehicle** | 115.2 [114.1, 116.3] | 115.5 [114.3, 116.8] | 116.0 [114.8, 117.2] | 203.4 | 203.4 | 203.4 |
| | **Road** | 116.0 [114.8, 117.2] | 116.3 [115.1, 117.6] | 116.2 [114.9, 117.4] | 203.4 | 203.4 | 203.4 |
| | **Full Frame** | 116.1 [114.8, 117.3] | 116.2 [115.0, 117.4] | 116.0 [114.8, 117.2] | 203.4 | 203.4 | 203.4 |
| **Burst $T_B=100$ ms** | **Pedestrian** | 120.3 [117.3, 123.4] | 120.1 [117.2, 123.0] | 119.5 [116.7, 122.2] | 203.4 | 203.4 | 203.4 |
| | **Vehicle** | 120.2 [117.8, 122.7] | 119.6 [117.1, 122.1] | 117.5 [115.1, 119.8] | 203.4 | 203.4 | 203.4 |
| | **Road** | 119.7 [117.1, 122.4] | 119.7 [117.0, 122.4] | 118.8 [116.2, 121.5] | 203.4 | 203.4 | 203.4 |
| | **Full Frame** | 119.9 [117.3, 122.6] | 119.6 [117.0, 122.2] | 118.9 [116.3, 121.5] | 203.4 | 203.4 | 203.4 |

---

## 4. Key Takeaways & Recommendations

1. **Small vs Large Objects**: Slicing prevents drop of the entire frame when a short channel glitch hits. However, because critical small objects like pedestrians occupy a very narrow vertical band (Slice 2 in $S=4$), horizontal slicing does not offer finer protection for pedestrians than for background; if a burst hits Slice 2, pedestrians are lost for that interval. Slicing with $S=8$ provides slightly finer vertical granularity, lowering pedestrian AoI by ~0.8 ms under severe burst fading.
2. **Lateral vs Center**: Horizontal slicing provides uniform horizontal coverage across left, center, and right portions of the visual field. Lateral regions inside the camera frame receive the exact same freshness benefit as central regions.
