# Smart Traffic Noise Prediction Model for Nairobi City

## Overview

This repository contains an implementation of the road traffic noise (RTN) prediction model described in the research paper *"A Smart Traffic Noise Prediction Model for Nairobi City Using Artificial Neural Networks"* by Mary Wambui, James Kamau, and Esther Nyambura (Technical University of Kenya and University of Nairobi, 2025).

The paper proposes a machine learning-based model using a Multi-Layer Perceptron (MLP) Artificial Neural Network (ANN) to forecast equivalent noise levels (Leq in dBA) in Nairobi, Kenya. It addresses urban noise pollution in rapidly growing African cities by leveraging traffic and environmental data. The model outperforms traditional statistical models like CoRTN and RLS-90, achieving a Mean Absolute Error (MAE) of 0.86 dBA and an R² of 0.93.

In this repository, I have implemented the core MLP ANN model in Python using PyTorch, based on the architecture and methodology outlined in the paper. This includes data preprocessing, model training, hyperparameter tuning via grid search, and evaluation. I've also included scripts for simulating predictions with sample data and a basic web dashboard prototype for real-time noise monitoring (using Flask or Streamlit).

## Key Features from the Paper

- **Data Collection**: Based on 504 samples from 42 noise hotspots in Nairobi (e.g., Uhuru Park, Thika Road), collected in July 2025. Inputs include vehicle counts (motorcycles, light, medium, heavy), speed, lanes, Passenger Car Units (PCU), and flow type (congested, periodic, fluid).
- **Model Architecture**:
  - Input Layer: 8 features (motorcycles, light vehicles, medium vehicles, heavy vehicles, speed, lanes, PCU, flow type).
  - Hidden Layers: Two layers with 25 and 50 neurons, using ReLU activation.
  - Output Layer: Predicted Leq (noise level in dBA).
  - Loss Function: Mean Squared Error (MSE).
  - Optimizer: Adam.
  - Training: 80/20 train-test split, 5-fold cross-validation.
- **Performance**: MAE: 0.86 dBA, RMSE: 1.11 dBA, R²: 0.93, Pearson correlation: 0.96.
- **Comparisons**: Superior to models like XGBoost, SVR, Random Forest, CoRTN (MAE: 5.0 dBA), and RLS-90 (MAE: 11.0 dBA).
- **Deployment**: The paper describes a web-based dashboard for real-time predictions; this repo includes a simple prototype.

## What I've Done in the Code

- **Implementation of the MLP Model**: Reproduced the ANN architecture in PyTorch, including the forward pass, training loop, and evaluation metrics (MAE, RMSE, R²).
- **Data Preprocessing**: Scripts to handle feature engineering, such as converting vehicle counts to PCU and categorizing flow types. Sample synthetic data is provided to mimic the paper's dataset (since original data isn't publicly available).
- **Hyperparameter Tuning**: Used scikit-learn's GridSearchCV for optimizing learning rate, batch size, epochs, etc.
- **Evaluation and Visualization**: Code to compute performance metrics, plot prediction errors (e.g., box plots), and visualize daily noise variations (similar to Figures 3-6 in the paper).
- **Web Dashboard Prototype**: A basic Flask/Streamlit app where users can input traffic parameters and get real-time Leq predictions. This demonstrates the deployment aspect mentioned in the paper.
- **Additional Enhancements**: 
  - Added support for loading custom datasets (CSV format).
  - Included Jupyter notebooks for step-by-step exploration.
  - Simulated real-time data feeds for testing.

This implementation serves as a starting point for reproducing the results or extending the model (e.g., adding variables like road surface type or weather, as recommended in the paper).

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn, Matplotlib
- Flask or Streamlit (for the dashboard)

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. **Train the Model**:
   ```
   python train_model.py --data_path sample_data.csv
   ```
   This trains the MLP on provided data and saves the model to `noise_predictor.pth`.

2. **Make Predictions**:
   ```
   python predict.py --model_path noise_predictor.pth --input "motorcycles=10,light=50,medium=10,heavy=5,speed=45,lanes=3,pcu=1500,flow=0"
   ```
   Outputs the predicted Leq.

3. **Run the Dashboard**:
   ```
   streamlit run dashboard.py
   ```
   Open in your browser to input values and view predictions interactively.

4. **Notebooks**:
   - `exploration.ipynb`: Data preprocessing and visualization.
   - `model_training.ipynb`: Full training and evaluation pipeline.

## Limitations and Future Work

- As noted in the paper, the model could be improved by incorporating seasonal variations, weather, or road surface types.
- Potential extensions: Integrate with real-time traffic APIs or deploy on a cloud platform for public use.

## References

- Original Paper: [Road Traffic Noise Prediction based on ANN (4).pdf](link-to-paper-if-available)
- For full citations, see the paper's references section.

If you find this useful, star the repo or contribute improvements! Contact me for questions.