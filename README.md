# 🩻 Xray-Detection  

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://xray-detector.streamlit.app/)  
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)  

An AI-powered web app that analyzes **chest X-ray images** and predicts whether they show abnormalities (e.g. pneumonia). Built using **PyTorch** for the deep learning model and **Streamlit** for deployment.  

---

## 🌐 Demo  

👉 [Live Web App](https://xray-detector.streamlit.app/)  
Upload your X-ray image and get instant predictions in your browser.  

---

## ✨ Features  

- 📤 Upload chest X-ray images (`.jpg`, `.png`)  
- 🤖 AI model predicts whether lungs are **Normal** or **Abnormal**  
- 📊 Displays prediction probabilities for transparency  
- 🎨 Simple & intuitive UI powered by Streamlit  
- 🧪 Includes a **sample X-ray image** (`test_xray.jpg`) for quick testing  

---

## 📂 Project Structure  

```
Xray-detection/
│
├── app.py             # Streamlit web app
├── test.py            # CLI/batch testing script
├── requirements.txt   # Dependencies
├── best.pt            # Trained PyTorch model weights
├── test_xray.jpg      # Example X-ray for testing
└── README.md          # Project documentation
```

---

## ⚡ Installation & Setup  

1. **Clone the repository**
   ```bash
   git clone https://github.com/likhilliki/Xray-detection.git
   cd Xray-detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # .\venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally**
   ```bash
   streamlit run app.py
   ```
   App will be available at: `http://localhost:8501`

---

## 🧠 Model Details  

- Framework: **PyTorch**  
- Input: Grayscale/Color chest X-ray images  
- Output: Binary classification (`Normal` / `Abnormal`)  
- Weights: `best.pt` (pretrained on a medical dataset)  
- Dataset: Public chest X-ray datasets (e.g. NIH ChestX-ray14 or similar)  

---

## 📦 Requirements  

- Python ≥ 3.8  
- PyTorch  
- Streamlit  
- Torchvision  
- Pillow, Numpy, etc.  

Install all via:  
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing  

Contributions are welcome! 🎉  

- Fork the repo  
- Create a feature branch: `git checkout -b feature-name`  
- Commit changes: `git commit -m 'Added new feature'`  
- Push branch: `git push origin feature-name`  
- Open a Pull Request  

---

## 📜 License  

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.  

---

## 👤 Author  

Developed by **[likhilliki](https://github.com/likhilliki)**  
For questions or suggestions, open an issue or reach out on GitHub.  

