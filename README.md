# Edutech_Data_Science_Task16
# Edutech Solution Data Science Internship

## Task 16: Model Deployment (Basic)

### Project Overview
This project demonstrates the basic deployment of a trained Machine Learning model. Building upon **Task 15 (NLP Text Classification on IMDB Reviews)**, the trained Naive Bayes Classifier and TF-IDF Vectorizer are serialized using **Pickle** and deployed as a fully functional web interface using **Streamlit**.

The application accepts user-generated movie reviews, preprocesses the textual text inputs, transforms them into numerical vectors, and performs dynamic evaluation to classify the overall sentiment.

---

### Project Architecture & Workflow
1. **Model Training (Task 15):** Trained a Multinomial Naive Bayes model using NLTK and Scikit-learn on the IMDB reviews dataset.
2. **Serialization (Pickle):** Exported the trained model architecture and vectorizer parameters to separate `.pkl` files to bridge the gap between training and real-world usage.
3. **Web Deployment (Task 16):** Built a local machine production server utilizing the Streamlit framework to collect real-time data inputs and render live classifications.

---

### Folder Structure
```text
├── app.py                  # Main web application script (UI and processing engine)
├── requirements.txt        # Automated module dependencies configurations
├── vectorizer.pkl          # Saved TF-IDF transformation assets via Pickle
└── naive_bayes_model .pkl  # Serialized production Naive Bayes Classifier via Pickle
```

---

### How to Setup and Run Locally

1. **Clone or Download the Project Folder:**
   Ensure all baseline components are located inside a unified environment folder directory.

2. **Install Required Package Dependencies:**
   Open your terminal in VS Code and execute the package installation script:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Local Web Production Server:**
   Trigger the main application file with the following framework execution command:
   ```bash
   streamlit run app.py
   ```
   The platform will automatically launch the browser engine interface at: `http://localhost:8501`

---

### Interview Questions Answered

* **What is an API?**
  An API (Application Programming Interface) is a secure communication engine that allows two separate software components or systems to interact and exchange metadata smoothly without user intervention.
  
* **What is the purpose of Pickle?**
  Pickle is used for flattening python object structures into a byte stream (Serialization) so they can be saved directly onto a hard disk asset and safely unpacked later (Deserialization) into production setups without needing code retraining.
