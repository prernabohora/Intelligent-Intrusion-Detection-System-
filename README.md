Title
Intelligent Intrusion Detection System for IoT (IIDS) – Using Isolation Forest, Autoencoder,
and Supervised Learning with XAI
Project Summary
This project is a software-based prototype of an AI-driven intrusion detection system tailored
for IoT environments such as defense networks. It leverages unsupervised learning
(Isolation Forest), deep learning (Autoencoder), and supervised machine learning (Random
Forest/XGBoost) to detect anomalies in IoT traffic. To improve trust and transparency,
Explainable AI (XAI) tools such as SHAP are integrated to explain why a traffic flow is
flagged as malicious. A Flask or Streamlit dashboard is used to simulate IoT traffic and
display real-time alerts.
Objective
The goal is to design and implement a system that can automatically learn normal IoT traffic
patterns, detect deviations caused by malicious activities (such as DDoS, spoofing, or
malware injection), and provide human-understandable explanations for the alerts. The final
outcome is a prototype dashboard demonstrating how such a system can strengthen cyber
defense in IoT-enabled military systems.
Scope
In scope:
● Software-only implementation using public datasets (CICIoT2023, Bot-IoT,
UNSW-NB15).
● Development of ML and DL models for anomaly detection.
● Visualization of alerts in real-time via a dashboard.
● Integration of explainability using SHAP/LIME.

Out of scope:
● Actual military deployment or integration with live IoT hardware.
● Full-scale Security Operations Center integration.

Technical Detail
● Datasets: CICIoT2023, Bot-IoT, UNSW-NB15 (labeled IoT traffic with normal and
malicious flows).
● Preprocessing: Feature extraction (packet size, flow duration, protocol type, rate),
normalization with StandardScaler or MinMaxScaler.

● Models:
○ Isolation Forest (unsupervised anomaly detection).
○ Autoencoder neural network (deep anomaly detection using reconstruction
error).
○ Random Forest or XGBoost (supervised classification when labels exist).
● Explainability: SHAP values to interpret which features influenced detection.
● Evaluation Metrics: Precision, Recall, F1-score, ROC-AUC.
● Deployment: Flask or Streamlit dashboard to simulate traffic and visualize alerts.

Implementation of Technology
1. Data preprocessing: Clean and scale traffic data from chosen datasets.
2. Isolation Forest: Train unsupervised model on normal data to detect anomalies.
3. Autoencoder: Train deep learning model on normal data; use reconstruction error to
detect malicious flows.
4. Supervised model: Train Random Forest/XGBoost when labeled datasets are
available for known attacks.
5. Explainability: Use SHAP to highlight top features contributing to anomalies.
6. Integration: Develop a lightweight dashboard to simulate IoT traffic and present
real-time alerts with explanations.
