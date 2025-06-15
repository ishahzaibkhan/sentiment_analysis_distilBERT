import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QGroupBox, QSizePolicy, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import pandas as pd

# Add these imports for transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class SentimentGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis UI")
        self.resize(940, 650)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.train_samples = []
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Title Label (first line)
        title = QLabel("Sentiment Analysis Model")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("background-color: #FFD700; border: 1px solid black;")
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(38)
        main_layout.addWidget(title)

        # Load Model Button (second line, full width)
        load_model_btn = QPushButton("Load Model")
        load_model_btn.setFixedHeight(32)
        load_model_btn.setStyleSheet("background-color: #B0C4DE;")
        load_model_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(load_model_btn)

        # Input Row (Search Line)
        input_layout = QHBoxLayout()
        input_label = QLabel("Enter Text For Sentiment:")
        input_label.setFixedWidth(250)
        input_layout.addWidget(input_label)

        input_edit = QLineEdit()
        input_edit.setText("")
        input_edit.setFixedHeight(28)
        input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(input_edit)
        main_layout.addLayout(input_layout)

        # Button Row (Load Dataset and Run Inference Manual, each 50% width)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(0)

        btn_load_data = QPushButton("Load Dataset")
        btn_load_data.setFixedHeight(32)
        btn_load_data.setStyleSheet("background-color: #B0C4DE;")
        btn_load_data.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_row.addWidget(btn_load_data, 1)

        btn_run_manual = QPushButton("Run Inference Manual")
        btn_run_manual.setFixedHeight(32)
        btn_run_manual.setStyleSheet("background-color: #B0C4DE;")
        btn_run_manual.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_row.addWidget(btn_run_manual, 1)

        main_layout.addLayout(btn_row)

        # Text Areas
        text_area_layout = QHBoxLayout()
        text_area_layout.setSpacing(10)

        # Left Table Area for Train Data
        left_box = QGroupBox("Train Data Samples:")
        left_layout = QVBoxLayout()
        self.left_table = QTableWidget()
        self.left_table.setColumnCount(1)
        # self.left_table.setHorizontalHeaderLabels(["Text"])  # Remove or comment out this line
        self.left_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.left_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.left_table.setSelectionMode(QTableWidget.SingleSelection)
        self.left_table.horizontalHeader().setStretchLastSection(True)
        self.left_table.horizontalHeader().setVisible(False)  # Add this line to hide the header
        self.left_table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.left_table)
        left_box.setLayout(left_layout)
        text_area_layout.addWidget(left_box)

        # Right Table Area for Test Data
        right_box = QGroupBox("Test Data Samples:")
        right_layout = QVBoxLayout()
        self.right_table = QTableWidget()
        self.right_table.setColumnCount(1)
        # self.right_table.setHorizontalHeaderLabels(["Text"])  # Remove or comment out this line
        self.right_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.right_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.right_table.setSelectionMode(QTableWidget.SingleSelection)
        self.right_table.horizontalHeader().setStretchLastSection(True)
        self.right_table.horizontalHeader().setVisible(False)
        self.right_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.right_table)
        right_box.setLayout(right_layout)
        text_area_layout.addWidget(right_box)

        main_layout.addLayout(text_area_layout)

        # Bottom Sentiment Labels
        bottom_layout = QHBoxLayout()
        true_sentiment = QLabel("True Sentiment: Neutral*")
        true_sentiment.setFont(QFont("Arial", 12, QFont.Bold))
        true_sentiment.setStyleSheet("background-color: #32CD32; color: white; padding: 6px;")
        true_sentiment.setAlignment(Qt.AlignLeft)
        true_sentiment.setFixedHeight(34)
        true_sentiment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bottom_layout.addWidget(true_sentiment)

        predicted_sentiment = QLabel("Predicted Sentiment: Neutral*")
        predicted_sentiment.setFont(QFont("Arial", 12, QFont.Bold))
        predicted_sentiment.setStyleSheet("background-color: #FF6347; color: white; padding: 6px;")
        predicted_sentiment.setAlignment(Qt.AlignLeft)
        predicted_sentiment.setFixedHeight(34)
        predicted_sentiment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bottom_layout.addWidget(predicted_sentiment)

        main_layout.addLayout(bottom_layout)

        # Save references for later use
        self.load_model_btn = load_model_btn
        self.btn_load_data = btn_load_data
        self.btn_run_manual = btn_run_manual
        self.input_edit = input_edit
        self.left_table = self.left_table  # for clarity
        self.right_table = self.right_table
        self.true_sentiment = true_sentiment
        self.predicted_sentiment = predicted_sentiment

        # Connect buttons
        self.load_model_btn.clicked.connect(self.load_model)
        self.btn_load_data.clicked.connect(self.load_dataset)
        self.btn_run_manual.clicked.connect(self.run_manual_inference)
        self.left_table.itemSelectionChanged.connect(self.on_train_sample_selected)
        self.right_table.itemSelectionChanged.connect(self.on_test_sample_selected)

    def load_model(self):
        try:
            model_dir = QFileDialog.getExistingDirectory(self, "Select Model Folder", "model")
            if not model_dir:
                return

            # Check for required files (update for safetensors and tokenizer files)
            required_files = [
                "config.json", "model.safetensors", "tokenizer_config.json",
                "vocab.txt", "tokenizer.json", "special_tokens_map.json"
            ]
            missing = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
            if missing:
                QMessageBox.critical(self, "Error", f"Missing files in model folder: {', '.join(missing)}")
                return

            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            QMessageBox.information(self, "Model Loaded", "BERT model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def load_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", "")
        if not folder:
            return
        train_path = os.path.join(folder, "train.csv")
        test_path = os.path.join(folder, "test.csv")
        try:
            # Load train.csv
            if os.path.isfile(train_path):
                df_train = pd.read_csv(train_path)
                if 'label' in df_train.columns:
                    train_samples = list(zip(df_train['text'], df_train['label']))
                else:
                    train_samples = [ (line.rsplit(',', 1)[0], line.rsplit(',', 1)[1]) for line in df_train['text'] ]
                self.train_samples = train_samples
                self.left_table.setRowCount(len(train_samples))
                for i, (text, label) in enumerate(train_samples):
                    self.left_table.setItem(i, 0, QTableWidgetItem(str(text)))
                self.left_table.resizeColumnsToContents()
            else:
                self.train_samples = []
                self.left_table.setRowCount(0)

            # Load test.csv
            if os.path.isfile(test_path):
                df_test = pd.read_csv(test_path)
                if 'label' in df_test.columns:
                    test_samples = list(zip(df_test['text'], df_test['label']))
                else:
                    test_samples = [ (line.rsplit(',', 1)[0], line.rsplit(',', 1)[1]) for line in df_test['text'] ]
                self.test_samples = test_samples
                self.right_table.setRowCount(len(test_samples))
                for i, (text, label) in enumerate(test_samples):
                    self.right_table.setItem(i, 0, QTableWidgetItem(str(text)))
                self.right_table.resizeColumnsToContents()
            else:
                self.test_samples = []
                self.right_table.setRowCount(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def on_train_sample_selected(self):
        selected = self.left_table.selectedItems()
        if not selected:
            self.true_sentiment.setText("True Sentiment: N/A")
            self.predicted_sentiment.setText("Predicted Sentiment: N/A")
            return
        row = self.left_table.currentRow()
        text = self.left_table.item(row, 0).text()
        true_label = self.train_samples[row][1]  # Get label from the loaded data
        self.true_sentiment.setText(f"True Sentiment: {self.map_label(true_label)}")
        pred = self.predict_sentiment(text)
        self.predicted_sentiment.setText(f"Predicted Sentiment: {self.map_label(pred)}")

    def on_test_sample_selected(self):
        selected = self.right_table.selectedItems()
        if not selected:
            self.true_sentiment.setText("True Sentiment: N/A")
            self.predicted_sentiment.setText("Predicted Sentiment: N/A")
            return
        row = self.right_table.currentRow()
        text = self.right_table.item(row, 0).text()
        true_label = self.test_samples[row][1]
        self.true_sentiment.setText(f"True Sentiment: {self.map_label(true_label)}")
        pred = self.predict_sentiment(text)
        self.predicted_sentiment.setText(f"Predicted Sentiment: {self.map_label(pred)}")

    def run_manual_inference(self):
        # Make sure this is the correct reference to the input box
        text = self.input_edit.text() if hasattr(self, 'input_edit') else ""
        if not text.strip():
            QMessageBox.warning(self, "Input Error", "Please enter text for sentiment prediction.")
            self.true_sentiment.setText("True Sentiment: N/A")
            self.predicted_sentiment.setText("Predicted Sentiment: N/A")
            return
        pred = self.predict_sentiment(text)
        self.true_sentiment.setText("True Sentiment: N/A")
        self.predicted_sentiment.setText(f"Predicted Sentiment: {self.map_label(pred)}")

    def predict_sentiment(self, text):
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        try:
            if not text.strip():
                return "No input"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model.eval()  # Ensure model is in eval mode
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Debug: print logits to check output
                # print("Logits:", logits.cpu().numpy())
                pred_label = torch.argmax(logits, dim=1).item()
            return pred_label
        except Exception as e:
            return f"Error: {e}"

    def map_label(self, label):
        # Map label index or string to sentiment
        label_map = {
            "0": "Negative", 0: "Negative",
            "1": "Neutral", 1: "Neutral",
            "2": "Positive", 2: "Positive"
        }
        return label_map.get(label, str(label))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SentimentGUI()
    gui.showMaximized()  # Start maximized and allow resizing
    sys.exit(app.exec_())