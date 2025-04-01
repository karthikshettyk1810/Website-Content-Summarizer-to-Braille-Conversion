"""
Enhanced summarization module using LLM-based approaches with newspaper3k content extraction.
This module provides improved summarization capabilities for the URL to Braille Converter.
"""

import os
import traceback
import re
import nltk
from newspaper import Article
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import openai
import pickle
import sys
import torch
import sentencepiece as spm
from functools import lru_cache
import logging
from pathlib import Path
import requests
import json
from dotenv import load_dotenv

# Set up logging for better diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_summarization')

# Configure Hugging Face cache directory
# Create a persistent cache directory in the project folder
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Using model cache directory: {CACHE_DIR}")

# Set environment variables for Hugging Face to use our cache directory
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(CACHE_DIR, 'hub')

# Load API keys from environment files
load_dotenv('openAi.env')
API_KEY = os.environ.get('OPENAI_API_KEY')

# Load Gemini API key from environment file
load_dotenv('gemini.env')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    logger.warning("Gemini API key not found in environment file. Create a file named 'gemini.env' with your API key in the format: GEMINI_API_KEY=your_api_key_here")

if not API_KEY:
    logger.warning("OpenAI API key not found in environment file, trying to load from file directly")
    try:
        with open('openAi.env', 'r') as f:
            content = f.read().strip()
            if content.startswith('OPENAI_API_KEY='):
                API_KEY = content.split('=', 1)[1].strip()
                logger.info(f"Found OpenAI API key: {API_KEY[:10]}...")
    except Exception as e:
        logger.error(f"Error reading OpenAI API key from file: {e}")

# Ensure the OpenAI API key is set in environment
if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
    logger.info(f"Successfully loaded OpenAI API key: {API_KEY[:10]}...")
else:
    logger.warning("Failed to load OpenAI API key")

def extract_content_with_newspaper(url):
    """
    Enhanced content extraction using newspaper3k library, which is specialized for news articles
    and blog content extraction.
    """
    try:
        logger.info(f"Extracting content with newspaper3k from {url}")
        article = Article(url)
        article.download()
        article.parse()
        
        # Get the main text content
        text = article.text
        
        # If text is too short, it might not have parsed correctly
        if not text or len(text.strip()) < 100:
            logger.warning("newspaper3k extraction yielded insufficient content, falling back to BeautifulSoup")
            return None
            
        logger.info(f"newspaper3k extracted {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting content with newspaper3k: {e}")
        traceback.print_exc()
        return None

def summarize_with_gemini(text, min_length=150, max_length=300):
    """
    Summarize text using Google's Gemini API with the Gemini Pro 1.5 model.
    This function generates comprehensive summaries using Google's state-of-the-art Gemini model.
    
    To use this function, you need to create a file named 'gemini.env' in your project directory
    with your Gemini API key in the format: GEMINI_API_KEY=your_api_key_here
    
    You can get a Gemini API key from: https://aistudio.google.com/app/apikey
    """
    try:
        # Check if Gemini API key is available
        if not GEMINI_API_KEY:
            logger.error("Gemini API key not found. Create a file named 'gemini.env' with your API key")
            return None
            
        logger.info("Initializing Gemini Pro 1.5-pro-latest summarization...")
        
        # Prepare the API request
        api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare prompt for comprehensive summary (12-15 lines)
        prompt = f"""Please provide a comprehensive summary of the following text in 12-15 lines. 
        Focus on capturing all key information and main points:
        
        {text}
        """
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 0.95,
                "maxOutputTokens": 800,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
        }
        
        # Make the API request
        logger.info("Sending request to Gemini Pro 1.5 API...")
        response = requests.post(
            f"{api_url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the summary from the response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        summary = parts[0]['text'].strip()
                        logger.info(f"Gemini Pro 1.5 summary length: {len(summary)} characters")
                        return summary
            
            logger.error(f"Unexpected response structure from Gemini API: {response_data}")
            return None
        else:
            # If Gemini Pro 1.5 fails, try with Gemini Pro as fallback
            if response.status_code == 400 and "gemini-1.5-pro" in response.text:
                logger.warning("Gemini Pro 1.5 not available, falling back to Gemini Pro")
                return summarize_with_gemini_pro(text, min_length, max_length)
            
            logger.error(f"Gemini API request failed with status code {response.status_code}: {response.text}")
            return None
        
    except Exception as e:
        logger.error(f"Error in Gemini Pro 1.5 summarization: {e}")
        traceback.print_exc()
        return None

def summarize_with_gemini_pro(text, min_length=150, max_length=300):
    """
    Fallback function to summarize text using Gemini Pro model if Pro 1.5 is not available.
    """
    try:
        logger.info("Initializing Gemini Pro summarization (fallback)...")
        
        # Prepare the API request
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare prompt for comprehensive summary (12-15 lines)
        prompt = f"""Please provide a comprehensive summary of the following text in 12-15 lines. 
        Focus on capturing all key information and main points:
        
        {text}
        """
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 0.95,
                "maxOutputTokens": 800,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
        }
        
        # Make the API request
        logger.info("Sending request to Gemini Pro API...")
        response = requests.post(
            f"{api_url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the summary from the response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        summary = parts[0]['text'].strip()
                        logger.info(f"Gemini Pro summary length: {len(summary)} characters")
                        return summary
            
            logger.error(f"Unexpected response structure from Gemini Pro API: {response_data}")
            return None
        else:
            logger.error(f"Gemini Pro API request failed with status code {response.status_code}: {response.text}")
            return None
        
    except Exception as e:
        logger.error(f"Error in Gemini Pro summarization: {e}")
        traceback.print_exc()
        return None

def summarize_with_gpt2(text, max_length=300):
    """
    Summarize text using the pre-trained GPT-2 model from Hugging Face.
    This is a local model that doesn't require an API key.
    Updated to generate longer summaries (12-15 lines).
    """
    try:
        logger.info("Loading GPT-2 model for summarization...")
        # Load pre-trained model and tokenizer
        model_name = "gpt2-medium"  # Using medium size for better performance
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add a summarization prompt to guide the model for longer summaries
        prompt_text = f"Please provide a comprehensive summary of the following text in 12-15 lines, capturing all important information and key points:\n\n{text}\n\nSummary:"
        
        # Tokenize and encode the input text with the prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
        
        logger.info("Generating summary with GPT-2...")
        # Generate summary with appropriate parameters for longer summaries
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=max_length, 
            num_beams=5, 
            length_penalty=1.0,  # Encourages longer summaries
            no_repeat_ngram_size=3,  # Prevents repetition
            early_stopping=True,
            temperature=0.8  # Slightly higher temperature for more detailed output
        )
        
        # Decode the generated summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the summary part (after the prompt)
        if "Summary:" in summary:
            summary = summary.split("Summary:")[1].strip()
        
        logger.info(f"GPT-2 summary length: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.error(f"Error in GPT-2 summarization: {e}")
        traceback.print_exc()
        return None

def summarize_with_gpt(text, max_tokens=800):
    """
    Summarize text using OpenAI's GPT-3.5-turbo model with a direct approach.
    Updated to generate longer summaries (12-15 lines).
    """
    try:
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("No OpenAI API key found in environment")
            return None
        
        logger.info(f"Using API key: {api_key[:10]}...")
        
        # Import OpenAI client
        from openai import OpenAI
        
        # Create client with the API key
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client created successfully")
        
        # Send request to OpenAI API with instructions for longer summary
        logger.info("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text accurately and comprehensively."},
                {"role": "user", "content": f"Summarize the following text in a detailed manner, providing a comprehensive summary of 12-15 lines that captures all important information and key points:\n\n{text}"}
            ],
            temperature=0.7,  # Slightly higher temperature for more detailed output
            max_tokens=max_tokens
        )
        
        # Extract and return the summary
        summary = response.choices[0].message.content.strip()
        logger.info(f"OpenAI summary length: {len(summary)} characters")
        logger.info(f"Tokens used: {response.usage.total_tokens}")
        return summary
        
    except Exception as e:
        logger.error(f"Error in OpenAI summarization: {e}")
        traceback.print_exc()
        return None

def get_model_and_tokenizer(model_name, tokenizer_class, model_class, use_auth_token=None):
    """
    Helper function to get model and tokenizer with persistent caching for better performance.
    Uses a dedicated cache directory to ensure models are properly cached between sessions.
    
    Args:
        model_name: Name of the model to load
        tokenizer_class: Class of the tokenizer to use
        model_class: Class of the model to use
        use_auth_token: Optional auth token for private models
        
    Returns:
        tuple: (tokenizer, model)
    """
    # Check in-memory cache first
    if model_name in _MODEL_CACHE:
        logger.info(f"Using in-memory cached model: {model_name}")
        return _MODEL_CACHE[model_name]
    
    # Check if model exists in cache directory
    model_path = os.path.join(CACHE_DIR, 'models--' + model_name.replace('/', '--'))
    if os.path.exists(model_path):
        logger.info(f"Found model in cache directory: {model_path}")
    else:
        logger.info(f"Model not found in cache, will download to: {model_path}")
    
    try:
        # Load tokenizer with explicit cache directory
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = tokenizer_class.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            use_auth_token=use_auth_token
        )
        
        # Load model with explicit cache directory
        logger.info(f"Loading model {model_name}...")
        model = model_class.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            use_auth_token=use_auth_token
        )
        
        # Store in memory cache
        _MODEL_CACHE[model_name] = (tokenizer, model)
        logger.info(f"Successfully loaded and cached {model_name}")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise

def integrated_summarize(url, text_content, post_process_function=None, extract_key_sentences_function=None):
    """
    Integrated summarization function that tries multiple approaches in sequence:
    1. Enhanced content extraction with newspaper3k
    2. Gemini summarization (Google's state-of-the-art model)
    3. GPT-2 summarization (alternative offline model)
    4. OpenAI GPT summarization if API key is available
    5. Falls back to the caller's methods if all LLM-based methods fail
    
    Parameters:
    - url: The URL to summarize
    - text_content: The extracted text content (fallback if newspaper3k fails)
    - post_process_function: Function to post-process summaries for coherence
    - extract_key_sentences_function: Function to extract key sentences as last resort
    
    Returns:
    - Either a successful summary or None to indicate caller should use their existing fallbacks
    """
    try:
        # First, try enhanced content extraction with newspaper3k
        enhanced_content = extract_content_with_newspaper(url)
        
        # If newspaper3k succeeded, use the enhanced content
        if enhanced_content:
            text_to_summarize = enhanced_content
            logger.info("Using enhanced content extraction from newspaper3k")
        else:
            # Otherwise fall back to the provided text_content
            text_to_summarize = text_content
            logger.info("Using fallback text content for summarization")
        
        # 1. First try with Gemini model (Google's state-of-the-art model)
        logger.info("Attempting Gemini summarization...")
        try:
            summary = summarize_with_gemini(text_to_summarize)
            if summary and len(summary) >= 150:  # Ensure summary is substantial
                logger.info("Successfully created summary with Gemini")
                if post_process_function:
                    summary = post_process_function(summary)
                return summary
        except Exception as gemini_e:
            logger.error(f"Gemini summarization failed: {gemini_e}")
        
        # 2. Try with GPT-2 model (alternative offline approach)
        logger.info("Attempting GPT-2 summarization...")
        try:
            summary = summarize_with_gpt2(text_to_summarize)
            if summary and len(summary) >= 150:  # Ensure summary is substantial
                logger.info("Successfully created summary with GPT-2")
                if post_process_function:
                    summary = post_process_function(summary)
                return summary
        except Exception as gpt2_e:
            logger.error(f"GPT-2 summarization failed: {gpt2_e}")
        
        # 3. Try with OpenAI if API key is available
        logger.info("Attempting OpenAI GPT summarization...")
        try:
            summary = summarize_with_gpt(text_to_summarize)
            if summary and len(summary) >= 150:  # Ensure summary is substantial
                logger.info("Successfully created summary with OpenAI GPT")
                if post_process_function:
                    summary = post_process_function(summary)
                return summary
        except Exception as gpt_e:
            logger.error(f"OpenAI GPT summarization failed: {gpt_e}")
        
        # 4. Call the provided fallback functions if available
        try:
            if 'key_functions' in sys.modules and 'summarize_text' in dir(sys.modules['key_functions']):
                from key_functions import summarize_text as fallback_summarize_text
                logger.info("Using fallback summarize_text function from key_functions module")
                return fallback_summarize_text(url, text_to_summarize, post_process_function)
        except Exception as fallback_e:
            logger.error(f"Error using fallback functions: {fallback_e}")
            
        # If we get here, all methods failed
        return None
        
    except Exception as e:
        logger.error(f"Error in integrated summarization: {e}")
        traceback.print_exc()
        return None
