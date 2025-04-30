# 📊 Sentiment Analysis of Amazon Product Reviews

This project performs sentiment analysis on Amazon product reviews using the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. The goal is to classify reviews as positive or negative based on their textual content.

## 📁 Dataset

- **Source:** [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Description:** This dataset consists of over 500,000 food reviews from Amazon, spanning more than 10 years up to October 2012. Each review includes:
  - Product and user information
  - Review text and summary
  - Helpfulness ratings
  - Overall score (1 to 5)

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Text Processing: `nltk`, `re`
  - Machine Learning: `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`
- **Modeling Techniques:**
  - Naive Bayes Classifier
  - Logistic Regression
  - Support Vector Machines (SVM)

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Yateesh12/sentiment-analysis-of-amazon-product-review.git
   cd sentiment-analysis-of-amazon-product-review
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**

   - Visit the [Amazon Fine Food Reviews dataset page](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
   - Click on "Download" to obtain the `Reviews.csv` file.
   - Place `Reviews.csv` in the root directory of the project.

## 📊 Project Structure

```plaintext
sentiment-analysis-of-amazon-product-review/
├── Reviews.csv
├── analysis.py
├── requirements.txt
└── README.md
```

## 🧪 Running the Analysis

Execute the sentiment analysis script:

```bash
python analysis.py
```

This script will:

- Load and preprocess the data
- Convert text data into numerical features using techniques like TF-IDF
- Train machine learning models to classify sentiments
- Evaluate model performance using metrics like accuracy, precision, recall, and F1-score
- Visualize results through confusion matrices and ROC curves
