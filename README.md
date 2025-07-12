# Brain Tumor Detection Web Application

A modern, AI-powered web application for detecting brain tumors from MRI scans. Upload an MRI image and receive an instant, detailed medical report with diagnosis, confidence, treatment suggestions, and more—all in a beautiful dark-themed interface.

---

## Features
- **AI-Powered Detection:** Upload MRI scans and get instant tumor detection and classification.
- **Medical Report Generation:** Automatically generates a detailed, human-readable report with diagnosis, severity, treatment, precautions, and specialist recommendations.
- **Modern UI:** Clean, responsive, and accessible dark-themed interface.
- **Image Preview:** See your MRI before submitting.
- **Download/Print Reports:** Download or print the generated medical report with a single click.
- **Secure Uploads:** User images are not tracked by git and are stored locally.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Brain_Tumor
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Model Files
- Place your trained model file (e.g., `brain_tumor_model.h5`) in the `models/` directory.
- Make sure your model and label files are not tracked by git (see `.gitignore`).

### 5. Run the Application
```bash
python main.py
```
- The app will be available at [http://localhost:5000](http://localhost:5000)

---

## Usage
1. Open the web app in your browser.
2. Upload an MRI scan image (JPG/PNG).
3. Click **Analyze Image**.
4. View the AI-generated diagnosis and medical report.
5. Download or print the report as needed.

---

## Project Structure
```
Brain_Tumor/
├── main.py                  # Main Flask app
├── requirements.txt         # Python dependencies
├── models/                  # Trained model files (not tracked by git)
├── report_generator/        # Report generation logic
├── static/                  # CSS and JS for UI
├── templates/               # HTML templates
├── uploads/                 # Uploaded MRI images (not tracked by git)
├── Datasets/                # (Optional) Data for training
├── venv/                    # Python virtual environment
└── .gitignore               # Git ignore rules
```

---

## Technologies Used
- **Python 3**
- **Flask** (web framework)
- **TensorFlow/Keras** or **PyTorch** (for model, as applicable)
- **HTML5, CSS3, JavaScript** (frontend)
- **Font Awesome** (icons)

---

## License
This project is for educational and research purposes. For commercial or clinical use, consult the authors and ensure compliance with medical regulations.

---

## Contributing
Pull requests and suggestions are welcome! Please open an issue or submit a PR.

---

## Contact
For questions or support, please contact the project maintainer. 