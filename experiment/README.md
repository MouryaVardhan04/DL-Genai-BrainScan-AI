# Brain Tumor Detection - Experimentation Tools

This folder contains tools and scripts for experimenting with brain tumor detection models, analyzing datasets, and testing different approaches.

## 📁 Files Overview

### 🔬 `model_experiment.py`
**Purpose**: Main experimentation script for training and evaluating brain tumor detection models.

**Features**:
- Complete CNN model training pipeline
- Data augmentation and preprocessing
- Model evaluation with metrics (accuracy, classification report, confusion matrix)
- Training history visualization
- Automatic model saving and result tracking
- Experiment results saved as JSON files

**Usage**:
```bash
python experiment/model_experiment.py
```

**Key Functions**:
- `BrainTumorExperiment`: Main class for model experimentation
- `load_and_preprocess_data()`: Load and preprocess MRI images
- `create_cnn_model()`: Build CNN architecture
- `train_model()`: Train model with callbacks
- `evaluate_model()`: Evaluate model performance
- `plot_training_history()`: Visualize training progress
- `plot_confusion_matrix()`: Create confusion matrix plots

### 📊 `data_analysis.py`
**Purpose**: Comprehensive dataset analysis and visualization tools.

**Features**:
- Dataset statistics and class distribution analysis
- Image properties analysis (size, intensity, contrast)
- Interactive visualizations using Plotly
- Dataset balance assessment
- Comprehensive analysis reports
- Multiple visualization types (bar charts, pie charts, histograms, scatter plots)

**Usage**:
```bash
python experiment/data_analysis.py
```

**Key Functions**:
- `BrainTumorDataAnalyzer`: Main class for data analysis
- `load_dataset_info()`: Load basic dataset information
- `analyze_image_properties()`: Analyze image characteristics
- `plot_class_distribution()`: Visualize class distribution
- `plot_image_properties()`: Analyze image properties
- `create_interactive_plots()`: Generate interactive visualizations
- `generate_report()`: Create comprehensive analysis report

## 🚀 Quick Start

### 1. Data Analysis
```bash
cd experiment
python data_analysis.py
```
This will:
- Analyze your dataset structure
- Generate visualizations
- Create a comprehensive report
- Save plots to the experiment folder

### 2. Model Experimentation
```bash
cd experiment
python model_experiment.py
```
This will:
- Load and preprocess your MRI images
- Train a CNN model
- Evaluate model performance
- Save training plots and results
- Save the best model to `models/` directory

## 📈 Output Files

### Data Analysis Outputs:
- `class_distribution.png`: Class distribution visualization
- `image_properties.png`: Image properties analysis
- `data_analysis_report.txt`: Comprehensive dataset report

### Model Experiment Outputs:
- `training_history.png`: Training accuracy and loss plots
- `confusion_matrix.png`: Model confusion matrix
- `results_YYYYMMDD_HHMMSS.json`: Detailed experiment results
- `models/brain_tumor_model.h5`: Best trained model

## 🔧 Configuration

### Data Path
Update the data path in both scripts if your dataset is located elsewhere:
```python
# In model_experiment.py
experiment = BrainTumorExperiment(data_path="your/dataset/path")

# In data_analysis.py
analyzer = BrainTumorDataAnalyzer(data_path="your/dataset/path")
```

### Model Parameters
Modify training parameters in `model_experiment.py`:
```python
# Training parameters
epochs = 50
batch_size = 32
validation_split = 0.2

# Model architecture
input_shape = (224, 224, 3)
num_classes = 4
```

## 📊 Dataset Structure

Expected dataset structure:
```
Datasets/
└── MRI Images/
    ├── glioma/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── meningioma/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── pituitary/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── no_tumor/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## 🎯 Experimentation Tips

### For Data Analysis:
1. **Start with data analysis** to understand your dataset
2. Check class balance and image properties
3. Use the generated report to identify potential issues
4. Consider data augmentation if dataset is imbalanced

### For Model Training:
1. **Begin with default parameters** to establish a baseline
2. **Experiment with different architectures** by modifying `create_cnn_model()`
3. **Try different data augmentation** strategies
4. **Monitor training history** to detect overfitting
5. **Use early stopping** to prevent overfitting
6. **Save experiment results** for comparison

### Hyperparameter Tuning:
```python
# Example: Try different learning rates
learning_rates = [0.001, 0.0001, 0.00001]
for lr in learning_rates:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), ...)

# Example: Try different batch sizes
batch_sizes = [16, 32, 64]
for batch_size in batch_sizes:
    experiment.run_experiment(batch_size=batch_size)
```

## 📝 Notes

- **Large datasets**: The scripts are designed to handle large datasets efficiently
- **Memory usage**: Monitor memory usage with large datasets
- **GPU usage**: Scripts will automatically use GPU if available
- **Results tracking**: All experiments are timestamped and saved
- **Reproducibility**: Set random seeds for reproducible results

## 🔍 Troubleshooting

### Common Issues:
1. **"No data found"**: Check your data path and dataset structure
2. **Memory errors**: Reduce batch size or image size
3. **Slow training**: Use GPU or reduce model complexity
4. **Poor accuracy**: Check data quality and try data augmentation

### Performance Tips:
- Use GPU for faster training
- Reduce image size for faster processing
- Use data generators for memory efficiency
- Monitor system resources during training

---

**Happy Experimenting! 🧠🔬** 