# Website Content Summarizer to Braille Conversion

A sophisticated web application that fetches content from URLs, summarizes it using advanced AI techniques, and converts it to braille representation with support for multiple languages including English, Hindi, and Kannada.

## Features

- **Advanced Content Extraction**: Uses newspaper3k and BeautifulSoup for intelligent content extraction from web pages
- **Multi-Model Summarization**: Implements a cascade of summarization techniques:
  - Google Gemini Pro 1.5 summarization
  - OpenAI GPT-3.5 summarization
  - Local GPT-2 model summarization
  - LSA (Latent Semantic Analysis) summarization
  - Fallback key sentence extraction
- **Multi-Language Support**:
  - Processes content in English, Hindi, Kannada, and mixed language texts
  - Automatic language detection
  - Translation capabilities
- **Braille Conversion**:
  - Standard English braille conversion
  - Devanagari/Bharati braille for Hindi
  - Kannada braille conversion
  - Mixed language braille handling
- **Audio Generation**: Text-to-speech functionality using Google TTS
- **Voice Command Correction**: AI-powered voice command correction using GPT-4
- **Caching System**: Redis-based caching for improved performance
- **Responsive Web Interface**: Modern, accessible UI

## Technical Architecture

### Core Components

1. **Content Extraction**

   - `extract_main_content()`: Extracts main content from webpages
   - `extract_content_with_newspaper()`: Enhanced extraction using newspaper3k
   - `clean_text()`: Removes citations, extra whitespace, and other noise

2. **Summarization Pipeline**

   - `integrated_summarize()`: Orchestrates multiple summarization approaches
   - `summarize_with_gemini()`: Uses Google's Gemini Pro 1.5 model
   - `summarize_with_gpt()`: Uses OpenAI's GPT-3.5-turbo model
   - `summarize_with_gpt2()`: Uses local Hugging Face GPT-2 model
   - `summarize_text()`: Uses LSA algorithm from Sumy
   - `extract_key_sentences()`: Fallback method using NLTK and frequency analysis
   - `post_process_summary()`: Ensures summaries form meaningful sentences
   - `ensure_coherence()`: Improves logical flow of summaries

3. **Language Processing**

   - `detect_language()`: Identifies the primary language of text
   - `translate_text()`: Translates content between languages
   - `process_mixed_language_summary()`: Handles text with multiple languages

4. **Braille Conversion**

   - `simple_braille_conversion()`: Standard English text to braille
   - `devanagari_braille_conversion()`: Hindi text to Bharati braille
   - `kannada_braille_conversion()`: Kannada text to Bharati braille
   - `mixed_language_braille_conversion()`: Handles text with multiple languages

5. **Audio Generation**

   - `generate_audio_google_tts()`: Creates audio using Google Translate TTS

6. **Voice Command Processing**

   - `correct_voice_command()`: AI-powered correction of voice commands

7. **Caching System**
   - Redis-based caching with automatic maintenance
   - `generate_cache_key()`: Creates unique keys for cached content
   - `clear_all_cache()`: Periodically clears cache to maintain freshness

### API Endpoints

- `/`: Main web interface
- `/process`: Processes URLs to extract, summarize, and convert content
- `/search`: Searches for topics and generates summaries
- `/translate`: Translates text between languages
- `/audio`: Generates audio from text
- `/voice-command`: Processes and corrects voice commands
- `/cache-status`: Returns Redis cache status
- `/clear-cache`: Manually clears the Redis cache

## Dependencies

The application relies on numerous libraries and frameworks:

### Web Framework

- Flask
- Flask-CORS
- Flask-SocketIO
- Gunicorn (for production deployment)

### Content Processing

- BeautifulSoup4
- newspaper3k
- requests
- lxml
- jusText

### Natural Language Processing

- NLTK
- spaCy
- Sumy
- transformers (Hugging Face)
- sentencepiece

### AI and Machine Learning

- OpenAI API
- Google Generative AI (Gemini)
- Hugging Face Transformers
- TensorFlow/PyTorch (backend for transformers)

### Translation and Language

- deep-translator
- googletrans
- langcodes
- pycountry

### Audio Processing

- gTTS (Google Text-to-Speech)
- PyAudio
- SpeechRecognition
- pydub

### Braille Support

- louis (optional for advanced braille translation)

### Caching and Performance

- Redis
- Flask-Caching

### Utilities

- python-dotenv
- fuzzywuzzy
- Levenshtein
- python-dateutil

## Installation

1. Clone this repository:

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Create `openAi.env` with your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - Create `gemini.env` with your Google Gemini API key: `GEMINI_API_KEY=your_key_here`
   - Create `REDIS_URL.env` with your Redis connection details (optional)

4. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

1. Start the Flask application:

   ```
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://127.0.0.1:5000/
   ```

3. Enter a URL in the input field and click "Process URL"

4. View the summarized content and its braille representation

## Memory Optimization

The application includes several memory optimization techniques for deployment environments:

- Transformers offline mode to avoid downloading models during runtime
- Disabled tokenizer parallelism
- Reduced TensorFlow logging
- Garbage collection after processing requests
- Redis cache management to prevent memory leaks

## Deployment

The application is configured for deployment on platforms like Render with:

- Procfile for Gunicorn configuration
- render.yaml for Render deployment settings
- Environment variable handling for API keys and configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
