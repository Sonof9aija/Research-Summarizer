# Research-Summarizer

This repository contains a Python application that allows users to summarize the contents of PDF files using natural language processing techniques.

## Features

- Extracts text from PDF files
- Summarizes the extracted text using TF-IDF and cosine similarity
- Generates a summarized PDF with both the original and summarized text

## Requirements

- Python 3.x
- NLTK
- scikit-learn
- NumPy
- NetworkX
- ReportLab
- PyPDF2
- Tkinter

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Sonof9aija/Research-Summarizer.git
    cd Research-Summarizer
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    python PDFsummarizer.py
    ```

2. Use the file browser to select a PDF file you want to summarize.

3. The application will process the file and generate a summarized PDF in the same directory as the original file with "_summarized" appended to the filename.

## File Structure

- `summarizer.py`: Main application file containing the logic for text extraction, summarization, and PDF generation.
- `requirements.txt`: List of required packages.

## Code Overview

### FileBrowserApp Class

Handles the UI and file browsing functionality using Tkinter.

### Summarizer Class

Handles text extraction, processing, and summarization.

## License

This project is licensed under the MIT License.

