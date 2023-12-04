# Ethereum-Future-Price-Prediction-(Time Series Prediction using LSTM)

* This repository contains code for a time series prediction model using Long Short-Term Memory (LSTM) networks implemented in Python with Keras and TensorFlow.

#Overview

1. Data Preprocessing:

* Loads time series data from a CSV file (ETH_1H.csv).
* Cleans the data by handling missing values and duplicates.
* Splits the data into training and testing sets.
* Creates additional time-related features.
  
2. Model Building:

* Constructs an LSTM-based neural network for time series prediction.
* Compiles the model using the Adam optimizer and Mean Squared Error (MSE) loss function.
* Trains the model on the training dataset.
  
3. Model Evaluation:

* Evaluates the trained model's performance on the testing dataset.
* Generates predictions and plots them against the actual values.

4. Future Prediction:

* Retrains the model on the entire dataset.
* Forecasts future values beyond the available data using the LSTM model.

5.Saving Model and Data:

* Saves the trained model (model.h5), MinMaxScaler (scaler.pkl), and processed data arrays (windows.npy, target.npy) for future use.

6. Deploy the model using flask web applications

#outcomes

1. Data Preprocessing:

* Loading and cleaning the time series data from ETH_1H.csv.
* Sorting the data based on the date.
* Creating new time-related features like hour, day, month, year, day of the week, day of the year, and week of the year.

2. Model Training and Evaluation:

* Building an LSTM-based neural network for time series prediction.
* Splitting the dataset into training and testing sets.
* Scaling the data using MinMaxScaler.
* Training the model on the training dataset for a specified number of epochs.
* Plotting the training and testing data to visualize the split.
* Evaluating the model's performance on the testing dataset using Mean Squared Error (MSE) loss and R-squared (coefficient of determination) metrics.
* Generating plots to visualize the actual vs. predicted values for the test dataset.

3.Future Prediction:

* Retraining the LSTM model on the entire dataset.
* Forecasting future values beyond the available data.
* Plotting the future predictions along with the original time series data.

4. Saving Trained Model and Data:

* Saving the trained model (model.h5) for future use.
* Saving the MinMaxScaler object (scaler.pkl) for inverse scaling of predictions.
*Storing processed data arrays (windows.npy, target.npy) to avoid reprocessing when reusing the data.

5. Potential Improvements:

* This code provides a foundational structure for time series prediction using LSTMs. However, further enhancements can be made to improve the model's accuracy, such as hyperparameter tuning, experimenting with different neural network architectures, considering additional relevant features, and augmenting the dataset if feasible.
