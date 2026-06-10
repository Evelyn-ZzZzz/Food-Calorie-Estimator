# Food-Calorie-Estimator

**An AI-powered food recognition and calorie estimation system using YOLOv8n**

Use YOLOv8n to identify food items and calculate total calories with high accuracy. This project combines computer vision and nutrition data to help users track their dietary intake.

---

## 🎯 Project Overview

Food-Calorie-Estimator leverages state-of-the-art deep learning models to:
- **Detect and classify** food items from images using YOLOv8n
- **Estimate portion sizes** and volumes using depth analysis
- **Calculate nutritional content** (calories, macronutrients)
- **Provide an interactive UI** for easy use

The system is trained on the **Nutrition5k dataset** (113 food classes) and includes a web-based interface for seamless interaction.

---

## 👥 Team & Contributions

- **Yihan Zhao** — YOLOv8n Model Training & Development
- **Wending Zhu** — UI Simulation & User Interface Design
- **Advik Iyer** — Project Idea Proposal
- **Chih-Jung Hsu** — Latency Optimization & Performance

---

## 📁 Project Structure

```
Food-Calorie-Estimator/
├── train.py              ← YOLOv8n training script
├── data.yaml             ← Dataset configuration (113 food classes)
├── yolov8n.pt            ← Pretrained model weights
│
├── AI glass/             ← Smart glasses UI application
│   ├── app.py            ← Gradio web interface
│   ├── run.bat           ← One-click launcher (Windows)
│   ├── requirements.txt  ← Python dependencies
│   │
│   ├── models/           
│   │   └── best.pt       ← Fine-tuned YOLOv8-seg weights
│   │
│   ├── core/             ← Analysis pipeline
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── stereo_depth.py      ← Depth estimation
│   │   ├── yolo_detect.py       ← Food detection
│   │   ├── volume_nutrition.py  ← Volume & calorie calculation
│   │   ├── nutrition_tables.py  ← Nutritional database
│   │   ├── llm_narrator.py      ← AI narration (optional)
│   │   ├── tts_speaker.py       ← Text-to-speech output
│   │   └── metrics_report.py    ← Performance metrics
│   │
│   └── samples/
│       ├── demo/         ← Bundled stereo image samples
│       └── stereo/       ← Calibration data
│
├── runs/                 ← Training outputs & checkpoints
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (Anaconda recommended)
- **NVIDIA GPU** (optional, for faster inference; CPU-only works too)
- **Windows 10/11** (for AI Glass UI)

### Installation

```bash
# Navigate to the AI Glass directory
cd "AI glass"

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Option A: Double-click launcher**
```
run.bat
```

**Option B: Command line**
```bash
python app.py
```

Then open your browser to **http://127.0.0.1:7860**

---

## 🎮 Using the UI

### Basic Workflow
1. **Load Images**: Upload left/right stereo images or use bundled demos
2. **Set Parameters**:
   - **B** (baseline): Distance between cameras in mm (~60 mm for iPhone)
   - **fx** (focal length): Camera focal length in pixels (~2852 for iPhone 1× portrait)
3. **Run Analysis**: Click to detect foods, estimate volume, and calculate calories
4. **View Results**: See detection overlay, depth map, AR nutrition panel, and narration

### Optional Features
- **DeepSeek AI Narration**: Requires API key in `.env`
- **Text-to-Speech**: Automatic voice output for results

---

## 🧠 Model Training

The YOLOv8n model is trained on the **Nutrition5k dataset** (113 food classes).

### Training Script
```python
python train.py
```

**Key Parameters:**
- Dataset: Nutrition5k (train/val/test split)
- Model: YOLOv8n (nano, fastest)
- Epochs: 100 (with early stopping after 20 epochs without improvement)
- Batch size: 16 (optimized for NVIDIA A100 GPU)
- Image size: 640×640

Training outputs are saved in the `runs/` directory.

---

## 🍎 Supported Foods

The model recognizes **113 food classes** including:
- Fruits & vegetables (apple, banana, bell pepper, avocado, etc.)
- Proteins (bacon, beans, almonds, etc.)
- Grains (bread, rice, pasta, etc.)
- Dairy & prepared foods
- And many more!

See `data.yaml` for the complete list.

---

## ⚙️ Key Configuration

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| **B** | Baseline (distance between cameras) | 60 mm (iPhone) |
| **fx** | Horizontal focal length in pixels | 2852 px (iPhone 1×) |
| **Epochs** | Training iterations | 100 |
| **Batch size** | Samples per training step | 16 |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 7860 already in use | Close previous `app.py` processes and restart |
| Missing `best.pt` | Ensure model weights are in `AI glass/models/` |
| Unrealistic calorie values | Verify **B** and **fx** parameters for accurate depth |
| No AI narration | Check `.env` file and DeepSeek API connection |
| No audio output | Check Windows Sound settings for default device |

---

## 📊 Performance

- **Inference Speed**: ~50-100ms per image (GPU)
- **CPU Mode**: ~300-500ms per image
- **Model Size**: ~7 MB (YOLOv8n)
- **Detection Classes**: 113 food types

---

## 🛠️ Technologies Used

- **Deep Learning**: [YOLOv8](https://github.com/ultralytics/ultralytics) (object detection)
- **Computer Vision**: OpenCV, stereo depth estimation
- **UI Framework**: Gradio (web interface)
- **Optional AI**: DeepSeek LLM for narration
- **Dataset**: Nutrition5k (113 food classes)

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

For questions or contributions, please reach out to the team members or open an issue on GitHub.

---

## 📚 References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Nutrition5k Dataset**: https://www.kaggle.com/datasets/sujaykapadwala/nutrition5k
- **Gradio**: https://gradio.app/

---

**Last Updated**: June 2026
```

Just copy this entire content and paste it directly into your README.md file on GitHub. You can use the edit button at the top right of the file view to make the changes!
