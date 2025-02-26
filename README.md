# Nepali PDF Assistant

A Streamlit app designed to process Nepali PDFs, answer questions, and provide semantic search capabilities.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Tags](#tags)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Extract Nepali (Devanagari) Text:**  
  Leverages advanced OCR techniques to accurately extract text from PDF files written in Nepali.

- **Answer Queries:**  
  Interact with the app in both English and Nepali. Ask questions and get intelligent responses based on the content.

- **Semantic Search:**  
  Quickly locate relevant information within documents through semantic search capabilities.

- **Conversation Tracking:**  
  Maintains your conversation history, allowing you to revisit previous queries and responses.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/nepali-pdf-assistant.git
Install Dependencies
Navigate to the project directory and run:

bash
Copy
Edit
pip install -r requirements.txt
Set Environment Variables
Create a .env file in the project root and add your Google API key:

bash
Copy
Edit
GOOGLE_API_KEY=your_key
Run the App
Start the Streamlit app by running:

bash
Copy
Edit
streamlit run app.py
Usage
Open your browser and go to http://localhost:8501.
Select your preferred language.
Upload a PDF file containing Nepali text.
Ask questions to interact with the document content.
Reset the session anytime to start fresh.
