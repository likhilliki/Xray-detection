# ðŸ©» AI-Powered X-ray Foreign Body Detector  

*â€œSee beyond the scan.â€*  
An AI-powered tool to detect foreign bodies (metal, glass, wood, etc.) in X-ray images using *YOLOv8 + OpenCV + Streamlit*.  

---

## ðŸš€ Features  
- âœ… Upload any X-ray image and get instant detection results  
- âœ… Bounding box visualization for detected foreign objects  
- âœ… Adjustable detection confidence threshold  
- âœ… Streamlit web app with clean UI  
- âœ… Future-ready for more medical anomaly detection (fractures, tumors, etc.)  

---

## ðŸ“‚ Project Structure  

foreign_body_detector/
â”‚â”€â”€ best.pt              # Trained YOLOv8 model
â”‚â”€â”€ app.py               # Streamlit app
â”‚â”€â”€ test.py              # Local testing script
â”‚â”€â”€ data.yaml            # Dataset configuration
â”‚â”€â”€ /train               # Training images (if included)
â”‚â”€â”€ /valid               # Validation images (if included)
â”‚â”€â”€ README.md            # Project documentation


---

## âš™ï¸ Installation  

### 1ï¸âƒ£ Clone the repo  
bash
git clone https://github.com/your-username/foreign-body-detector.git
cd foreign-body-detector


### 2ï¸âƒ£ Create a virtual environment (Anaconda recommended)  
bash
conda create -n xray_env python=3.10 -y
conda activate xray_env


### 3ï¸âƒ£ Install dependencies  
bash
pip install ultralytics opencv-python streamlit matplotlib


---

## ðŸ“Š Usage  

### ðŸ”¹ Run Streamlit app  
bash
streamlit run app.py


### ðŸ”¹ Run local test script  
bash
python test.py


Upload an X-ray image â†’ the app highlights detected foreign bodies with bounding boxes.  

---

## ðŸ“¸ Demo Screenshots  
(Add images of your app running here)  

---

## ðŸ§  Model Training  
- Labeled dataset prepared using *Roboflow*  
- Trained with *YOLOv8*  
- Classes:  
  - 0: No foreign body  
  - 1: Foreign body  

To retrain:  
bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640


---

## ðŸŒ Future Scope  
- Extend detection to fractures, tumors, lung diseases  
- Mobile app integration  
- Deployment in hospitals and rural clinics  

---

## ðŸ¤ Contributors  
ðŸ‘¨â€ðŸ’» Your Name(s)  

---

## ðŸ“œ License  
This project is licensed under the MIT License.
