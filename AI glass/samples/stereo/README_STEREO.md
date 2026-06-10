# Stereo test image pairs (with known B and fx)

There is **no widely used public dataset** of “food + rectified left/right + calories” with **B and fx printed on every photo**. Use one of these instead:

## 1. Middlebury Stereo 2021 (recommended for pipeline debug)

- **Download:** https://vision.middlebury.edu/stereo/data/scenes2021/
- Each scene folder has `im0.png` (left), `im1.png` (right), `calib.txt`.
- From `calib.txt` (example scene):
  - **baseline** = 111.53 mm → set **B = 111.53**
  - **f** from `cam0` matrix ≈ **1758.23 px** → set **fx = 1758**
  - **width** = 1920 (use full resolution; do not resize before depth)

**Note:** Scenes are general indoor/objects, not food. Use this to verify **depth + volume math**, then switch to your food cameras.

Place files here:

```
stereo_samples/middlebury/SCENE1/im0.png
stereo_samples/middlebury/SCENE1/im1.png
```

Then in the UI click **Load Middlebury pair** (if present).

## 2. PlantStereo (plants, not food; has stereo + calibration code)

- https://github.com/wangqingyu985/PlantStereo  
- Left/right + disparity; ZED baseline/focal constants in their scripts.

## 3. Google Nutrition5k (food, but not side-by-side stereo pairs)

- Overhead **RealSense RGB-D**, not two horizontal eye cameras.  
- https://github.com/google-research-datasets/Nutrition5k  

## 4. Your iPhone 17 Pro Max (1× main camera)

- **4284 × 5712**, 24 mm equivalent → **fx ≈ 2856** (portrait width 4284).  
- **B** = physical distance between your two prototype cameras (mm), **not** from the phone EXIF.

## 5. Nutrition5k test set (YOLO only)

- Good for **detection**; default **B=65, fx=500** is wrong for full-res → **volume will explode** unless you use **bbox fallback** (now automatic when stereo fails).
