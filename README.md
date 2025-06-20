# Sentiment Analysis UI

## Overview
This project is a graphical user interface (GUI) for sentiment analysis using a pre-trained DistilBERT model. The application allows users to load a dataset, select training or test examples, and manually input text for sentiment prediction. The sentiment is classified into three categories: **Negative**, **Neutral**, and **Positive**.

## Features
- **Load Model**: Load a pre-trained DistilBERT model and tokenizer from a folder containing configuration files.
- **Load Dataset**: Load training and test datasets (`train.csv` and `test.csv`) from a selected folder.
- **Training Table**: Displays all training examples in a scrollable table. Selecting a row predicts its sentiment and displays the true and predicted sentiment at the bottom.
- **Test Table**: Displays all test examples in a scrollable table with the same functionality as the training table.
- **Manual Inference**: Allows users to input custom text and predict its sentiment using the loaded model.

## Requirements
- Python 3.8 or higher
- Libraries:
  - `PyQt5`
  - `transformers`
  - `torch`
  - `pandas`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the required Python libraries:
   ```bash
   pip install PyQt5 transformers torch pandas
   ```

## Usage
1. Run the application:
   ```bash
   python gui.py
   ```
2. **Load Model**:
   - Click the "Load Model" button.
   - Select the folder containing the model files (`config.json`, `model.safetensors`, `tokenizer_config.json`, etc.).
3. **Load Dataset**:
   - Click the "Load Dataset" button.
   - Select the folder containing `train.csv` and `test.csv`.
   - Training examples will appear in the left table, and test examples will appear in the right table.
4. **Select Examples**:
   - Click on a row in the training or test table to predict its sentiment.
   - The true sentiment and predicted sentiment will appear at the bottom.
5. **Manual Inference**:
   - Enter custom text in the input box.
   - Click "Run Inference Manual" to predict its sentiment.

## Dataset Format
The application expects the dataset files (`train.csv` and `test.csv`) to have the following format:
- **Columns**:
  - `text`: The text for sentiment analysis.
  - `label`: The sentiment label (0=Negative, 1=Neutral, 2=Positive).

Example:
```csv
text,label
The company reported a huge loss this quarter.,0
The meeting is scheduled for next week.,1
The company achieved record profits this year.,2
```

## Model Files
The model folder should contain the following files:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer_config.json`
- `tokenizer.json`
- `vocab.txt`
- `special_tokens_map.json`

## Troubleshooting
- **Model not loaded**: Ensure the model folder contains all required files.
- **Neutral sentiment not predicted**: Check the training data and model compatibility.
- **Dataset not loaded**: Ensure `train.csv` and `test.csv` are in the selected folder.

## License
This project is licensed under the MIT License.

## Author
Shahzaib Khan