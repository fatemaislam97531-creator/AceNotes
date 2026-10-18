# 🎓 AceNotes

AceNotes helps students study faster. It uses **Gemma 4**, an open-weight model, through the Gemini API.

## Features
- **Clean My Notes:** upload photos of messy handwritten notes and get clean, organized notes.
- **Textbook → Quiz:** upload a photo of a textbook page and get a summary, key terms, and an interactive quiz.
- **Quiz from your notes:** turn your cleaned notes into a quiz with one click.
- English and Bangla output.
- Download your notes and quizzes as Markdown.

## Why it matters
Many students study from handwritten notes and printed books. AceNotes saves time and helps them test themselves. Bangla support makes it useful for students in Bangladesh.

## Run it
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here      # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
streamlit run app.py
```
Get a free API key at https://aistudio.google.com

## Models used
- `gemma-4-31b-it` (default, best quality)
- `gemma-4-26b-a4b-it` (faster)

## License
MIT
