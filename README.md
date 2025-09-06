# Handwriting Classification (Human vs. Computer-Generated)

This project is a web application that uses a Convolutional Neural Network (CNN) to classify an image of a word as either human-written or computer-generated. The model is trained on a dataset of handwritten words and a custom-generated dataset of words rendered with various fonts.

## Features

-   Generates a dataset of computer-written text images using various fonts.
-   Preprocesses both human and computer-written image data.
-   Builds and trains a CNN model using TensorFlow and Keras.
-   Provides a simple web interface using Flask to upload an image and get a prediction.

## Project Structure

```
.
├── app.py                                  # Main Flask application file
├── Human Vs Computer Handwriting...ipynb   # Jupyter Notebook for data prep and model training
├── Word_Prediction.keras                   # The trained and saved Keras model
├── requirements.txt                        # Required Python packages
├── README.md                               # This file
├── FONTS/                                  # Directory for font files (.ttf, .otf)
├── hand_words/                             # Directory for the human handwriting dataset
├── templates/
│   └── index.html                          # HTML template for the web app
└── uploads/                                # Default folder for uploaded images
```

## Setup and Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed on your system.

2.  **Clone the Repository**: If this were a git repository, you would clone it. For now, just ensure you have all the project files in one directory.

3.  **Install Dependencies**: Open your terminal or command prompt in the project directory and install the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```


## Usage

There are two main parts to this project: retraining the model and running the prediction web application.

### 1. Running the Web Application

To start the web server and use the pre-trained model for predictions:

1.  Open your terminal in the project directory.

2.  Set the Flask application environment variable. 
    
    On Windows:
    ```cmd
    set FLASK_APP=app.py
    ```
    On macOS/Linux:
    ```bash
    export FLASK_APP=app.py
    ```

3.  Run the Flask application.
    ```bash
    flask run
    ```

4.  Open your web browser and navigate to `http://127.0.0.1:5000`.

5.  Upload an image of a word to classify it.

### 2. Retraining the Model

If you want to modify the model or retrain it on new data, use the Jupyter Notebook:

1.  Make sure you have Jupyter Notebook installed (`pip install notebook`).
2.  Run Jupyter Notebook from your terminal:
    ```bash
    jupyter notebook
    ```
3.  Open the `Human Vs Computer Handwriting Classification Project Final Khushboo.ipynb` file.
4.  You can run the cells sequentially to perform the following steps:
    -   Generate computer-written text images.
    -   Preprocess all image data.
    -   Split the data into training, validation, and test sets.
    -   Build and compile the CNN model.
    -   Train the model and save the new `Word_Prediction.keras` file.

## Technologies Used

-   **Backend**: Python, Flask
-   **Machine Learning**: TensorFlow, Keras, Scikit-learn
-   **Data Processing**: NumPy, Pillow (PIL), OpenCV, Scikit-image
-   **Data Visualization**: Matplotlib, Seaborn

## Screenshots

### Web App Page
![Computer Generated ](screenshot/Computer_Generated_Prediction/.png)
![Computer Generated ](screenshot/Human_Written_Prediction/.png)



