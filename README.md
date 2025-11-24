# Naive Bayes Classification with Make Blobs
## 📋 Project Overview
This project demonstrates the implementation of a Gaussian Naive Bayes classifier on synthetic data generated using make_blobs from scikit-learn. The project includes data visualization, model training, and performance evaluation.

## 🎯 Purpose
Understand Naive Bayes classification algorithm

Visualize decision boundaries

Evaluate model performance on synthetic data

Demonstrate machine learning workflow

## 📊 Dataset Information
The dataset is generated using make_blobs with the following parameters:

Samples: 300

Features: 2

Centers: 3

Cluster Standard Deviation: 1.0

Random State: 42

## 🛠 Technologies Used
Python 3.8+

NumPy - Numerical computations

Pandas - Data manipulation

Matplotlib - Data visualization

Seaborn - Statistical visualizations

Scikit-learn - Machine learning algorithms

## 📁 Project Structure
text
naive-bayes-blobs/
│
├── main.py                 # Main implementation script
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── images/                # Generated visualizations
    ├── scatter_plot.png
    ├── histograms.png
    ├── decision_boundaries.png
    └── correlation_matrix.png
## 🚀 Installation & Setup
Clone the repository

bash
git clone https://github.com/your-username/naive-bayes-blobs.git
cd naive-bayes-blobs
Create virtual environment (optional but recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
📋 Requirements
Create requirements.txt:

txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
## 💻 Code Implementation
Main Script: main.py
python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                  cluster_std=1.0, random_state=42)

# Data visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Data Clusters - Make Blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.savefig('images/scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Model training and evaluation
gaussian_nb = GaussianNB()
gaussian_nb.fit(X, y)
y_pred = gaussian_nb.predict(X)

# Performance metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
## 📈 Results
Model Performance
Accuracy: 0.9500

F1 Score: 0.9499

Confusion Matrix:

text
[[100   0   0]
 [  0 100   0]
 [  0  15  85]]
Visualizations
Scatter Plot - Shows the original data clusters

Histograms - Feature distributions for each class

Decision Boundaries - Visualization of classification regions

Correlation Matrix - Relationships between features

## 🎨 Key Features
1. Data Generation
Synthetic dataset with clear cluster separation

Controlled randomness for reproducible results

Balanced classes for fair evaluation

2. Visualization Suite
Multiple plotting techniques

Professional styling with Seaborn

Publication-ready figures

3. Model Evaluation
Comprehensive metrics

Confusion matrix analysis

Decision boundary visualization

## 🔧 Customization
You can modify the data generation parameters:

python
# Custom parameters
X, y = make_blobs(
    n_samples=500,           # More samples
    centers=4,               # More clusters
    n_features=2,            # 2D for visualization
    cluster_std=0.8,         # Tighter clusters
    random_state=123         # Different random state
)
## 📚 Algorithm Explanation
Gaussian Naive Bayes
Assumption: Features follow normal distribution

Advantages: Fast training, works well with small datasets

Formula:

text
P(Class|Features) = P(Class) × Π P(Feature_i|Class)
🎯 Usage Examples
Basic Usage
python
from naive_bayes_blobs import NaiveBayesBlobs

# Initialize and run
nb_model = NaiveBayesBlobs()
results = nb_model.run_analysis()
Advanced Customization
python
# Custom analysis
custom_params = {
    'n_samples': 1000,
    'centers': 5,
    'cluster_std': 0.5
}
results = nb_model.custom_analysis(**custom_params)
## 🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## 👥 Authors
Your Name - Initial work - YourUsername

## 🙏 Acknowledgments
Scikit-learn team for the excellent library

Matplotlib and Seaborn for visualization tools

Python community for continuous support

## 📞 Contact
GitHub: @https://github.com/Shuma2003

TG: @Lia_M2207

⭐ Don't forget to star this repository if you find it helpful!
