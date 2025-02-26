Nepali PDF Assistant
A Streamlit app designed to process Nepali PDFs, answer questions, and provide semantic search capabilities.

What It Can Do
Extract Nepali (Devanagari) Text: The app uses advanced OCR techniques to extract text from PDF files.

Answer Queries: Responds to questions in both English and Nepali.

Semantic Search: Enables users to search document content semantically.

Conversation Tracking: Keeps track of your conversation history.

Installation
Clone the Repository:

bash
git clone https://github.com/yourusername/nepali-pdf-assistant.git
Install Dependencies:

bash
pip install -r requirements.txt
Set Environment Variables:
Create a .env file and add your Google API key:

bash
GOOGLE_API_KEY=your_key
Run the App:

bash
streamlit run app.py
Usage
Open the app at http://localhost:8501.

Select your preferred language.

Upload a PDF file.

Ask questions and explore the document content.

Reset the session anytime.