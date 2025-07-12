"""
Brain Tumor Dataset Analysis Script
This script analyzes the brain tumor dataset, provides statistics, and creates visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BrainTumorDataAnalyzer:
    def __init__(self, data_path="Datasets/MRI Images"):
        self.data_path = data_path
        self.dataset_stats = {}
        self.image_data = []
        
    def load_dataset_info(self):
        """Load basic information about the dataset"""
        print("Loading dataset information...")
        
        tumor_types = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        dataset_info = {}
        
        for tumor_type in tumor_types:
            tumor_path = os.path.join(self.data_path, tumor_type)
            if os.path.exists(tumor_path):
                files = [f for f in os.listdir(tumor_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                dataset_info[tumor_type] = {
                    'count': len(files),
                    'files': files[:10],  # Store first 10 filenames for reference
                    'path': tumor_path
                }
            else:
                dataset_info[tumor_type] = {'count': 0, 'files': [], 'path': tumor_path}
        
        self.dataset_stats = dataset_info
        return dataset_info
    
    def analyze_image_properties(self, sample_size=50):
        """Analyze properties of sample images"""
        print("Analyzing image properties...")
        
        image_properties = []
        
        for tumor_type, info in self.dataset_stats.items():
            if info['count'] > 0:
                tumor_path = info['path']
                files = [f for f in os.listdir(tumor_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Sample images for analysis
                sample_files = files[:min(sample_size, len(files))]
                
                for img_file in sample_files:
                    img_path = os.path.join(tumor_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            height, width, channels = img.shape
                            
                            # Calculate basic statistics
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mean_intensity = np.mean(gray)
                            std_intensity = np.std(gray)
                            min_intensity = np.min(gray)
                            max_intensity = np.max(gray)
                            
                            image_properties.append({
                                'tumor_type': tumor_type,
                                'filename': img_file,
                                'width': width,
                                'height': height,
                                'channels': channels,
                                'mean_intensity': mean_intensity,
                                'std_intensity': std_intensity,
                                'min_intensity': min_intensity,
                                'max_intensity': max_intensity,
                                'contrast': max_intensity - min_intensity
                            })
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        self.image_data = image_properties
        return image_properties
    
    def create_dataset_summary(self):
        """Create a comprehensive dataset summary"""
        print("Creating dataset summary...")
        
        summary = {
            'total_images': sum(info['count'] for info in self.dataset_stats.values()),
            'tumor_types': list(self.dataset_stats.keys()),
            'class_distribution': {k: v['count'] for k, v in self.dataset_stats.items()},
            'dataset_balance': self._calculate_balance(),
            'image_properties': self._summarize_image_properties()
        }
        
        return summary
    
    def _calculate_balance(self):
        """Calculate dataset balance metrics"""
        counts = [info['count'] for info in self.dataset_stats.values()]
        total = sum(counts)
        
        if total == 0:
            return {'balanced': False, 'balance_ratio': 0}
        
        min_count = min(counts)
        max_count = max(counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        return {
            'balanced': balance_ratio > 0.7,  # Consider balanced if ratio > 0.7
            'balance_ratio': balance_ratio,
            'min_class_count': min_count,
            'max_class_count': max_count
        }
    
    def _summarize_image_properties(self):
        """Summarize image properties"""
        if not self.image_data:
            return {}
        
        df = pd.DataFrame(self.image_data)
        
        summary = {
            'avg_width': df['width'].mean(),
            'avg_height': df['height'].mean(),
            'avg_channels': df['channels'].mean(),
            'avg_intensity': df['mean_intensity'].mean(),
            'avg_contrast': df['contrast'].mean(),
            'size_variation': {
                'width_std': df['width'].std(),
                'height_std': df['height'].std()
            }
        }
        
        return summary
    
    def plot_class_distribution(self, save_path="experiment/class_distribution.png"):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        classes = list(self.dataset_stats.keys())
        counts = [self.dataset_stats[cls]['count'] for cls in classes]
        
        bars = ax1.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Brain Tumor Dataset Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tumor Type')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_image_properties(self, save_path="experiment/image_properties.png"):
        """Plot image properties analysis"""
        if not self.image_data:
            print("No image data available. Run analyze_image_properties() first.")
            return
        
        df = pd.DataFrame(self.image_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Brain Tumor Image Properties Analysis', fontsize=16, fontweight='bold')
        
        # Image sizes
        axes[0, 0].scatter(df['width'], df['height'], c=df['tumor_type'].astype('category').cat.codes, 
                           alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Dimensions')
        
        # Intensity distribution
        for tumor_type in df['tumor_type'].unique():
            subset = df[df['tumor_type'] == tumor_type]
            axes[0, 1].hist(subset['mean_intensity'], alpha=0.6, label=tumor_type, bins=20)
        axes[0, 1].set_xlabel('Mean Intensity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Intensity Distribution')
        axes[0, 1].legend()
        
        # Contrast by class
        df.boxplot(column='contrast', by='tumor_type', ax=axes[0, 2])
        axes[0, 2].set_title('Contrast by Tumor Type')
        axes[0, 2].set_xlabel('Tumor Type')
        axes[0, 2].set_ylabel('Contrast')
        
        # Standard deviation of intensity
        for tumor_type in df['tumor_type'].unique():
            subset = df[df['tumor_type'] == tumor_type]
            axes[1, 0].hist(subset['std_intensity'], alpha=0.6, label=tumor_type, bins=20)
        axes[1, 0].set_xlabel('Standard Deviation of Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Intensity Variability')
        axes[1, 0].legend()
        
        # Size distribution
        axes[1, 1].hist(df['width'], alpha=0.7, label='Width', bins=20)
        axes[1, 1].hist(df['height'], alpha=0.7, label='Height', bins=20)
        axes[1, 1].set_xlabel('Pixels')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Image Size Distribution')
        axes[1, 1].legend()
        
        # Aspect ratio
        df['aspect_ratio'] = df['width'] / df['height']
        for tumor_type in df['tumor_type'].unique():
            subset = df[df['tumor_type'] == tumor_type]
            axes[1, 2].hist(subset['aspect_ratio'], alpha=0.6, label=tumor_type, bins=20)
        axes[1, 2].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Aspect Ratio Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        if not self.image_data:
            print("No image data available. Run analyze_image_properties() first.")
            return
        
        df = pd.DataFrame(self.image_data)
        
        # Interactive scatter plot of image dimensions
        fig1 = px.scatter(df, x='width', y='height', color='tumor_type',
                          title='Brain Tumor Image Dimensions by Type',
                          labels={'width': 'Width (pixels)', 'height': 'Height (pixels)'})
        fig1.show()
        
        # Interactive box plot of intensity by tumor type
        fig2 = px.box(df, x='tumor_type', y='mean_intensity',
                      title='Mean Intensity Distribution by Tumor Type',
                      labels={'tumor_type': 'Tumor Type', 'mean_intensity': 'Mean Intensity'})
        fig2.show()
        
        # Interactive histogram of contrast
        fig3 = px.histogram(df, x='contrast', color='tumor_type',
                           title='Contrast Distribution by Tumor Type',
                           labels={'contrast': 'Contrast', 'count': 'Count'})
        fig3.show()
    
    def generate_report(self, save_path="experiment/data_analysis_report.txt"):
        """Generate a comprehensive data analysis report"""
        print("Generating data analysis report...")
        
        summary = self.create_dataset_summary()
        
        report = f"""
BRAIN TUMOR DATASET ANALYSIS REPORT
===================================

Dataset Overview:
----------------
Total Images: {summary['total_images']}
Tumor Types: {', '.join(summary['tumor_types'])}

Class Distribution:
------------------
"""
        
        for tumor_type, count in summary['class_distribution'].items():
            percentage = (count / summary['total_images']) * 100 if summary['total_images'] > 0 else 0
            report += f"{tumor_type.capitalize()}: {count} images ({percentage:.1f}%)\n"
        
        report += f"""

Dataset Balance:
---------------
Balanced: {'Yes' if summary['dataset_balance']['balanced'] else 'No'}
Balance Ratio: {summary['dataset_balance']['balance_ratio']:.3f}
Min Class Count: {summary['dataset_balance']['min_class_count']}
Max Class Count: {summary['dataset_balance']['max_class_count']}

Image Properties Summary:
------------------------
"""
        
        if summary['image_properties']:
            props = summary['image_properties']
            report += f"""Average Width: {props['avg_width']:.1f} pixels
Average Height: {props['avg_height']:.1f} pixels
Average Channels: {props['avg_channels']:.1f}
Average Intensity: {props['avg_intensity']:.1f}
Average Contrast: {props['avg_contrast']:.1f}
Width Standard Deviation: {props['size_variation']['width_std']:.1f}
Height Standard Deviation: {props['size_variation']['height_std']:.1f}

Recommendations:
---------------
"""
        
        # Add recommendations based on analysis
        if summary['dataset_balance']['balance_ratio'] < 0.7:
            report += "- Dataset is imbalanced. Consider data augmentation or resampling techniques.\n"
        
        if summary['image_properties']:
            if props['size_variation']['width_std'] > 50 or props['size_variation']['height_std'] > 50:
                report += "- High variation in image sizes. Consider standardizing image dimensions.\n"
            
            if props['avg_intensity'] < 50 or props['avg_intensity'] > 200:
                report += "- Unusual intensity values. Consider normalization techniques.\n"
        
        report += "- Consider using data augmentation to improve model generalization.\n"
        report += "- Implement proper train/validation/test splits for model evaluation.\n"
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {save_path}")
        print(report)
    
    def run_complete_analysis(self):
        """Run complete data analysis pipeline"""
        print("Starting Brain Tumor Dataset Analysis")
        print("=" * 50)
        
        # Load dataset info
        self.load_dataset_info()
        
        # Analyze image properties
        self.analyze_image_properties()
        
        # Create visualizations
        self.plot_class_distribution()
        self.plot_image_properties()
        
        # Generate report
        self.generate_report()
        
        print("Data analysis completed!")

def main():
    """Main function to run the data analysis"""
    analyzer = BrainTumorDataAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 