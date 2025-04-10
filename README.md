# Drug Classification Project

## Overview

The **Drug Classification Project** is a machine learning application designed to predict the appropriate drug category for patients based on their medical attributes. By leveraging patient data such as age, sex, blood pressure levels, cholesterol levels, and the sodium-to-potassium ratio, the model aims to assist healthcare professionals in making informed prescribing decisions.



## Dataset

The dataset used in this project comprises patient information with the following features:

- **Age:** Patient's age in years.
- **Sex:** Gender of the patient ('M' for Male, 'F' for Female).
- **Blood Pressure:** Categorized as 'HIGH', 'NORMAL', or 'LOW'.
- **Cholesterol:** Levels indicated as 'HIGH' or 'NORMAL'.
- **Na_to_K Ratio:** The sodium-to-potassium ratio in the patient's blood.
- **Drug Type:** The prescribed drug category (target variable).

*Note: Ensure the dataset (`data.csv`) is placed in the `data/` directory.*



### Prerequisites

- Python 3.10 or higher
  

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JISNAp/drug-classification.git
   cd drug-classification
   ```
2. **Setup virtual environment**
   ```bash
   conda create -p ./venv python=3.10 -y
   conda activate ./venv
     ```
3. **Install requirements**
   ```bash
   pip install -r requirements.txt
     ```
4. **Train the model**
   ```bash
   python train.py
     ```
5. **Web application**
   ```bash
   python ./App/app.py
     ```
   Access the application at http://localhost:7860 in your web browser.

   
