import os
import requests
import traceback
import base64
import tempfile
import json
import urllib.parse
import urllib3
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
import re
from dotenv import load_dotenv
import redis
from urllib.parse import quote
import time
import random
import hashlib
import pickle
import concurrent.futures
from newspaper import Article
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import openai
import threading
import datetime

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment file
print("Loading OpenAI API key from openAi.env file...")
load_dotenv('openAi.env')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("API key not found in environment after loading .env file, trying to read directly")
    try:
        with open('openAi.env', 'r') as f:
            content = f.read().strip()
            if content.startswith('OPENAI_API_KEY='):
                OPENAI_API_KEY = content.split('=', 1)[1]
                print(f"Found OpenAI API key in file: {OPENAI_API_KEY[:10]}...")
    except Exception as e:
        print(f"Error reading OpenAI API key from file: {e}")

# Set the OpenAI API key
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    print(f"OpenAI API key set successfully: {OPENAI_API_KEY[:10]}...")
else:
    print("WARNING: OpenAI API key not found, LLM-based summarization will not be available")

# Import the enhanced summarization module
from enhanced_summarization import extract_content_with_newspaper, summarize_with_gpt, integrated_summarize
# Import key functions for fallback
from key_functions import summarize_text, extract_key_sentences

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
load_dotenv()

# Redis configuration
redis_host = 'redis-10473.c330.asia-south1-1.gce.redns.redis-cloud.com'
redis_port = 10473
redis_username = 'default'
redis_password = 'lFNtMxdxllRsePZnSWt99xyL99YJR0vE'
REDIS_AVAILABLE = False
redis_client = None
last_cache_clear_time = None

try:
    # Connect to Redis using direct credentials instead of URL
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        username=redis_username,
        password=redis_password,
        decode_responses=False
    )
    redis_client.ping()  # Test the connection
    REDIS_AVAILABLE = True
    print("Redis Cloud connected successfully")
    
    # Set the initial cache clear time
    last_cache_clear_time = datetime.datetime.now()
    print(f"Initial cache clear time set to: {last_cache_clear_time}")
except Exception as e:
    print(f"Redis Cloud connection failed: {e}")
    REDIS_AVAILABLE = False

def generate_cache_key(prefix, data):
    """Generate a unique cache key based on the input data."""
    # For URL processing, use a simpler key based on URL and language
    if prefix == 'process' and 'url' in data:
        url = data['url'].lower()
        language = data.get('language', 'en')
        # Create a more direct key for URLs
        return f"{prefix}:{language}:{hashlib.md5(url.encode('utf-8')).hexdigest()}"
    
    # For other data types, use the full serialization
    serialized = json.dumps(data, sort_keys=True).encode('utf-8')
    return f"{prefix}:{hashlib.md5(serialized).hexdigest()}"

def check_redis_connection():
    """Check if Redis connection is working and return status."""
    if not REDIS_AVAILABLE:
        return False
    
    try:
        redis_client.ping()
        return True
    except Exception:
        return False

def clear_all_cache():
    """Clear all Redis cache entries."""
    global last_cache_clear_time
    
    if not REDIS_AVAILABLE:
        print("Redis cache is not available, skipping cache clear")
        return False
    
    try:
        # Clear all keys
        redis_client.flushdb()
        last_cache_clear_time = datetime.datetime.now()
        print(f"All cache entries cleared at {last_cache_clear_time}")
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def check_and_clear_cache_if_needed():
    """Check if cache needs to be cleared (once per day) and clear it if necessary."""
    global last_cache_clear_time
    
    if not REDIS_AVAILABLE or not last_cache_clear_time:
        return
    
    current_time = datetime.datetime.now()
    time_difference = current_time - last_cache_clear_time
    
    # Clear cache if it's been more than 24 hours since the last clear
    if time_difference.total_seconds() > 24 * 60 * 60:  # 24 hours in seconds
        print(f"Cache is older than 24 hours (last cleared: {last_cache_clear_time}), clearing now...")
        clear_all_cache()
        print("Automatic cache clearing completed")

# Start a background thread to periodically check and clear the cache
def start_cache_maintenance():
    """Start a background thread to periodically check and clear the cache."""
    def cache_maintenance_worker():
        while True:
            try:
                check_and_clear_cache_if_needed()
            except Exception as e:
                print(f"Error in cache maintenance: {e}")
            
            # Check every hour
            time.sleep(60 * 60)  # 60 minutes * 60 seconds
    
    if REDIS_AVAILABLE:
        # Clear cache on startup to ensure fresh results with new models
        clear_all_cache()
        
        # Start the maintenance thread
        maintenance_thread = threading.Thread(target=cache_maintenance_worker, daemon=True)
        maintenance_thread.start()
        print("Cache maintenance thread started")

@app.route('/cache-status', methods=['GET'])
def cache_status():
    """Return the status of the Redis cache."""
    redis_working = check_redis_connection()
    
    if redis_working:
        # Get some stats about the cache
        try:
            info = redis_client.info()
            keys = redis_client.keys('*')
            process_keys = [k for k in keys if k.decode('utf-8', errors='ignore').startswith('process:')]
            translate_keys = [k for k in keys if k.decode('utf-8', errors='ignore').startswith('translate:')]
            audio_keys = [k for k in keys if k.decode('utf-8', errors='ignore').startswith('audio:')]
            
            return jsonify({
                'status': 'connected',
                'total_keys': len(keys),
                'process_cache_entries': len(process_keys),
                'translation_cache_entries': len(translate_keys),
                'audio_cache_entries': len(audio_keys),
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory_human': info.get('used_memory_human', 'unknown')
            })
        except Exception as e:
            return jsonify({
                'status': 'connected',
                'error_getting_stats': str(e)
            })
    else:
        return jsonify({
            'status': 'disconnected',
            'message': 'Redis cache is not available'
        })

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the Redis cache."""
    global last_cache_clear_time
    
    if not REDIS_AVAILABLE:
        return jsonify({'status': 'error', 'message': 'Redis cache is not available'}), 400
    
    try:
        # Get the type of cache to clear
        data = request.json or {}
        cache_type = data.get('type', 'all')
        
        if cache_type == 'all':
            # Clear all keys
            redis_client.flushdb()
            last_cache_clear_time = datetime.datetime.now()
            return jsonify({'status': 'success', 'message': 'All cache entries cleared'})
        elif cache_type in ['process', 'translate', 'audio']:
            # Clear only specific type of keys
            keys = redis_client.keys(f'{cache_type}:*')
            if keys:
                redis_client.delete(*keys)
            return jsonify({'status': 'success', 'message': f'{len(keys)} {cache_type} cache entries cleared'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid cache type specified'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error clearing cache: {str(e)}'}), 500     

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_topic():
    data = request.json
    topic = data.get('topic')
    logger.debug(f"Search request received for topic: {topic}")
    
    if not topic:
        logger.warning("No topic provided in search request")
        return jsonify({'error': 'No topic provided'}), 400

    try:
        # Prepare results list - limit to just two results to make it faster
        results = []
        logger.debug(f"Starting search for topic: {topic}")
        
        # First try Wikipedia
        # Format the topic properly for Wikipedia URL (capitalize first letter of each word)
        formatted_topic = ' '.join(word.capitalize() for word in topic.split())
        logger.debug(f"Formatted topic for Wikipedia: {formatted_topic}")
        wiki_result = {
            'title': f'Wikipedia: {formatted_topic}',
            'link': f'https://en.wikipedia.org/wiki/{requests.utils.quote(formatted_topic.replace(" ", "_"))}',
            'snippet': f'Wikipedia encyclopedia article about {formatted_topic}',
            'source': 'Wikipedia',
            'icon': 'bi-book-fill',
            'color': '#3498db'
        }
        
        # Then YouTube
        youtube_result = {
            'title': f'YouTube: {topic}',
            'link': f'https://www.youtube.com/results?search_query={requests.utils.quote(topic)}',
            'snippet': f'YouTube videos about {topic}',
            'source': 'YouTube',
            'icon': 'bi-youtube',
            'color': '#e74c3c'
        }
        
        # Use threading to make both requests in parallel
        import threading
        
        def get_wikipedia_result():
            nonlocal wiki_result
            try:
                # First try direct Wikipedia API to get the exact article URL
                # Use proper search API to find the most relevant article for the topic
                search_api_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(topic)}&format=json&srlimit=1"
                logger.debug(f"Fetching Wikipedia search API: {search_api_url}")
                search_response = requests.get(search_api_url, timeout=5)
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    search_results = search_data.get('query', {}).get('search', [])
                    logger.debug(f"Wikipedia search results count: {len(search_results)}")
                    
                    if search_results:
                        # Get the exact page title from search results
                        exact_title = search_results[0].get('title')
                        
                        # Now get the full page data using the exact title
                        page_api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=info&titles={requests.utils.quote(exact_title)}&format=json&redirects=1"
                        page_response = requests.get(page_api_url, timeout=3)
                        
                        if page_response.status_code == 200:
                            page_data = page_response.json()
                            pages = page_data.get('query', {}).get('pages', {})
                            
                            # Check if we got a valid page
                            for page_id, page_info in pages.items():
                                if page_id != '-1':  # Valid page found
                                    page_title = page_info.get('title', exact_title)
                                    normalized_title = page_title.replace(' ', '_')
                                    exact_wiki_url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(normalized_title)}"
                                    
                                    # Get a snippet from the search results
                                    snippet = search_results[0].get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                                    if not snippet:
                                        snippet = f"Wikipedia article about {page_title}"
                                    
                                    # Update wiki result with exact URL
                                    wiki_result = {
                                        'title': f"Wikipedia: {page_title}",
                                        'link': exact_wiki_url,
                                        'snippet': snippet,
                                        'source': 'Wikipedia',
                                        'icon': 'bi-book-fill',
                                        'color': '#3498db'
                                    }
                                    logger.debug(f"Found exact Wikipedia article: {exact_wiki_url}")
                                    return  # We found the exact article, no need for Google search
                
                # If we reach here, the search API didn't find a good match, try the original method
                wiki_api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={requests.utils.quote(topic)}&redirects=1"
                wiki_api_response = requests.get(wiki_api_url, timeout=3)
                
                if wiki_api_response.status_code == 200:
                    wiki_data = wiki_api_response.json()
                    pages = wiki_data.get('query', {}).get('pages', {})
                    
                    # Check if we got a valid page (not a -1 missing page)
                    for page_id, page_data in pages.items():
                        if page_id != '-1':  # Valid page found
                            page_title = page_data.get('title', topic)
                            normalized_title = page_title.replace(' ', '_')
                            exact_wiki_url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(normalized_title)}"
                            
                            # Update wiki result with exact URL
                            wiki_result = {
                                'title': f"Wikipedia: {page_title}",
                                'link': exact_wiki_url,
                                'snippet': f"Wikipedia article about {page_title}",
                                'source': 'Wikipedia',
                                'icon': 'bi-book-fill',
                                'color': '#3498db'
                            }
                            return  # We found the exact article, no need for Google search
                
                # Fallback to Google search if API doesn't return a valid page
                wiki_query = f"https://www.google.com/search?q=site:wikipedia.org+{requests.utils.quote(topic)}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                # Add more robust error handling and logging
                logger.debug(f"Attempting to fetch URL: {wiki_query}")
                try:
                    response = requests.get(wiki_query, headers=headers, timeout=15, verify=False)
                    response.raise_for_status()
                    logger.debug(f"Successfully fetched URL: {wiki_query}, status: {response.status_code}")
                except requests.exceptions.SSLError as ssl_err:
                    logger.warning(f"SSL Error, retrying without verification: {ssl_err}")
                    response = requests.get(wiki_query, headers=headers, timeout=15, verify=False)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching URL: {e}")
                    
                    # Try one more time with a different User-Agent
                    try:
                        alt_headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299'
                        }
                        logger.debug("Retrying with alternative User-Agent")
                        response = requests.get(wiki_query, headers=alt_headers, timeout=15, verify=False)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as retry_err:
                        logger.error(f"Error on retry: {retry_err}")
                        return jsonify({'error': 'Failed to fetch URL content after multiple attempts', 'details': str(e)}), 400
            except Exception as e:
                logger.error(f"Error finding Wikipedia article: {str(e)}", exc_info=True)
                # Keep the default wiki result
        
        def get_youtube_result():
            nonlocal youtube_result
            try:
                youtube_query = f"https://www.google.com/search?q=site:youtube.com+{requests.utils.quote(topic)}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                # Add more robust error handling and logging
                logger.debug(f"Attempting to fetch URL: {youtube_query}")
                try:
                    response = requests.get(youtube_query, headers=headers, timeout=15, verify=False)
                    response.raise_for_status()
                    logger.debug(f"Successfully fetched URL: {youtube_query}, status: {response.status_code}")
                except requests.exceptions.SSLError as ssl_err:
                    logger.warning(f"SSL Error, retrying without verification: {ssl_err}")
                    response = requests.get(youtube_query, headers=headers, timeout=15, verify=False)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching URL: {e}")
                    
                    # Try one more time with a different User-Agent
                    try:
                        alt_headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299'
                        }
                        logger.debug("Retrying with alternative User-Agent")
                        response = requests.get(youtube_query, headers=alt_headers, timeout=15, verify=False)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as retry_err:
                        logger.error(f"Error on retry: {retry_err}")
                        return jsonify({'error': 'Failed to fetch URL content after multiple attempts', 'details': str(e)}), 400
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    youtube_soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find first YouTube result
                    youtube_result_div = youtube_soup.find('div', {'class': ['g', 'tF2Cxc']})
                    if youtube_result_div:
                        title_element = youtube_result_div.find('h3')
                        link_element = youtube_result_div.find('a')
                        snippet_element = youtube_result_div.find('div', {'class': ['VwiC3b', 'yXK7lf']})
                        
                        if title_element and link_element:
                            title = title_element.text
                            link = link_element['href']
                            if link.startswith('/url?'):
                                link = link.split('&sa=')[0].replace('/url?q=', '')
                            
                            snippet = snippet_element.text if snippet_element else "YouTube video about " + topic
                            
                            # Update the youtube result
                            youtube_result = {
                                'title': title,
                                'link': link,
                                'snippet': snippet,
                                'source': 'YouTube',
                                'icon': 'bi-youtube',
                                'color': '#e74c3c'
                            }
            except Exception as e:
                logger.error(f"Error finding YouTube video: {str(e)}", exc_info=True)
                # Keep the default youtube result
        
        # Start both threads
        wiki_thread = threading.Thread(target=get_wikipedia_result)
        youtube_thread = threading.Thread(target=get_youtube_result)
        
        wiki_thread.start()
        youtube_thread.start()
        
        # Wait for both threads with a timeout (max 4 seconds)
        wiki_thread.join(timeout=4)
        youtube_thread.join(timeout=4)
        
        # Add results to the list
        results.append(wiki_result)
        results.append(youtube_result)
        
        logger.debug(f"Search completed for topic: {topic}, returning {len(results)} results")
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f"Search failed: {str(e)}",
            'results': [
                {
                    'title': f'Wikipedia: {topic}',
                    'link': f'https://en.wikipedia.org/wiki/{requests.utils.quote(topic.replace(" ", "_"))}',
                    'snippet': 'Search Wikipedia for this topic',
                    'source': 'Wikipedia',
                    'icon': 'bi-book-fill',
                    'color': '#3498db'
                },
                {
                    'title': f'YouTube Search: {topic}',
                    'link': f'https://www.youtube.com/results?search_query={requests.utils.quote(topic)}',
                    'snippet': 'Find videos about this topic on YouTube',
                    'source': 'YouTube',
                    'icon': 'bi-youtube',
                    'color': '#e74c3c'
                }
            ]
        }), 200  # Return 200 with fallback result

@app.route('/process', methods=['POST'])
def process_url():
    """Process a URL to extract, summarize, and convert content to braille."""
    try:
        start_time = time.time()
        data = request.json
        url = data.get('url')
        target_language = data.get('language', 'en')
        
        logger.debug(f"Process request received for URL: {url}, language: {target_language}")

        if not url:
            logger.warning("No URL provided in process request")
            return jsonify({'error': 'No URL provided'}), 400

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            logger.debug(f"URL format corrected to: {url}")

        # Check cache first
        if REDIS_AVAILABLE:
            cache_key = generate_cache_key('process', {'url': url, 'language': target_language})
            logger.debug(f"Checking cache with key: {cache_key}")
            
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for URL: {url}, language: {target_language}")
                    
                    # Always regenerate audio data even for cached results
                    # This ensures fresh audio for each request
                    if target_language in ['en', 'hi', 'kn']:
                        text_for_audio = pickle.loads(cached_result).get('translated_summary') if target_language != 'en' and pickle.loads(cached_result).get('translated_summary') else pickle.loads(cached_result).get('summary')
                        if text_for_audio:
                            pickle.loads(cached_result)['audio'] = generate_audio_google_tts(text_for_audio, target_language)
                    
                    # Add cache hit indicator and processing time 
                    pickle.loads(cached_result)['cache_hit'] = True
                    pickle.loads(cached_result)['processing_time'] = 0  # No processing time for cache hits
                    return jsonify(pickle.loads(cached_result))
            except Exception as cache_error:
                logger.error(f"Error retrieving from cache: {str(cache_error)}", exc_info=True)
                # Continue with normal processing if cache retrieval fails
        
        logger.info(f"Processing URL: {url}")
        
        try:
            # Extract content from the URL
            logger.debug(f"Attempting to extract content using newspaper3k from: {url}")
            article_content = extract_content_with_newspaper(url)
            
            if not article_content or len(article_content.strip()) < 100:
                logger.debug("newspaper3k extraction yielded insufficient content, trying fallback method")
                # Fallback to our custom extraction method
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }, verify=False)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                article_content = extract_main_content(soup)
                title = soup.title.string if soup.title else "No title found"
            else:
                # If using newspaper3k, we need to get the title separately
                try:
                    response = requests.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }, verify=False)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else "No title found"
                except:
                    title = "No title found"
            
            logger.debug(f"Content extraction successful, content length: {len(article_content)}")
            
            # Clean the text
            cleaned_content = clean_text(article_content)
            
            # Ensure we have enough content to summarize
            if not cleaned_content or len(cleaned_content) < 200:
                logger.warning(f"Not enough content to summarize from URL: {url}")
                return jsonify({'error': 'Not enough content to summarize from this URL'}), 400
            
            # Try the enhanced LLM-based summarization approach first
            try:
                from enhanced_summarization import integrated_summarize
                from key_functions import summarize_text, extract_key_sentences
                logger.debug("Attempting enhanced LLM-based summarization...")
                summary = integrated_summarize(url, cleaned_content, post_process_summary, extract_key_sentences)
                
                if not summary:
                    logger.debug("Enhanced summarization failed, falling back to traditional methods")
                    summary = summarize_text(url, cleaned_content, post_process_summary)
            except Exception as e:
                logger.error(f"Error using enhanced summarization: {e}, falling back to original method")
                # Use the original summarization method
                summary = summarize_text(url, cleaned_content, post_process_summary)
            
            if not summary or len(summary) < 50:
                logger.debug("Summary generation failed, using extracted content")
                # Fallback to just the extracted content or a portion of it
                summary = cleaned_content[:500] + "..."
            
            # Apply post-processing to improve readability
            summary = post_process_summary(summary)
            summary = ensure_coherence(summary)
            
            # Detect language of the summary
            summary_language = detect_language(summary)
            
            # If the summary is in a mixed language, process it accordingly
            if summary_language == 'mixed':
                summary = process_mixed_language_summary(summary)
            
            # Translate if needed
            translated_summary = None
            if target_language != 'en':
                translated_summary = translate_text(summary, target_language)
            
            # Convert to braille based on language
            if target_language == 'hi':
                if translated_summary:
                    # Process Hindi text with potential English words properly
                    # Instead of filtering out non-Devanagari characters, we'll convert them too
                    braille = mixed_language_braille_conversion(translated_summary, target_language)
                else:
                    braille = "Translation failed, braille conversion not possible."
            elif target_language == 'kn':
                if translated_summary:
                    # Process Kannada text with potential English words properly
                    braille = mixed_language_braille_conversion(translated_summary, target_language)
                else:
                    braille = "Translation failed, braille conversion not possible."
            else:
                # Default to English braille
                braille = simple_braille_conversion(summary)

            # Generate audio
            audio_data = None
            if target_language in ['en', 'hi', 'kn']:
                text_for_audio = translated_summary if target_language != 'en' and translated_summary else summary
                audio_data = generate_audio_google_tts(text_for_audio, target_language)
            
            # Prepare the result
            result = {
                'original_url': url,
                'title': title,
                'summary': summary,
                'translated_summary': translated_summary,
                'braille': braille,
                'audio': audio_data,
                'language': target_language,
                'cache_hit': False,
                'processing_time': round(time.time() - start_time, 2)
            }
            
            # Cache the result if Redis is available
            if REDIS_AVAILABLE:
                try:
                    # Create a safe-to-cache copy without potentially problematic objects
                    cache_result = {
                        'original_url': url,
                        'title': str(title),
                        'summary': str(summary),
                        'translated_summary': str(translated_summary) if translated_summary else None,
                        'braille': str(braille),
                        'audio': audio_data,  # Binary data should be fine for pickle
                        'language': target_language,
                        'cache_hit': False,
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    
                    # Use a 24-hour cache expiration
                    redis_client.setex(
                        cache_key,
                        86400,  # 24 hours in seconds
                        pickle.dumps(cache_result)
                    )
                    logger.info(f"Cached result for URL: {url}")
                except Exception as e:
                    logger.error(f"Error caching result: {e}")
                    traceback.print_exc()  # Print the full stack trace for debugging
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            traceback.print_exc()
            return jsonify({'error': 'Failed to process URL', 'details': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error in process_url: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process URL request', 'details': str(e)}), 500

def translate_text(text, target_language):
    """Translate text using a free translation API with caching."""
    try:
        # Check cache first
        if REDIS_AVAILABLE:
            cache_key = generate_cache_key('translate', {'text': text, 'lang': target_language})
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Using cached translation for language: {target_language}")
                return pickle.loads(cached_result)
        
        # Existing translation code
        url = "https://translate.googleapis.com/translate_a/single"
        
        params = {
            "client": "gtx",
            "sl": "en",  # Source language (English)
            "tl": target_language,  # Target language
            "dt": "t",  # Return translated text
            "q": text  # Text to translate
        }
        
        # Make the request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract the translated text
        translated_text = ""
        for sentence in result[0]:
            if sentence[0]:
                translated_text += sentence[0]
        
        # Cache the result
        if REDIS_AVAILABLE and translated_text:
            # Cache translations for 7 days (604800 seconds)
            redis_client.setex(cache_key, 604800, pickle.dumps(translated_text))
            logger.info(f"Cached translation for language: {target_language}")
        
        return translated_text
    except Exception as e:
        logger.error(f"Translation API error: {e}")
        traceback.print_exc()
        return None

def extract_main_content(soup):
    """Extract the main content from the webpage, focusing on paragraphs and headings."""
    try:
        # Look for main content containers
        main_content = soup.find(['main', 'article', 'div', 'section'], class_=lambda c: c and any(x in str(c).lower() for x in ['content', 'article', 'main', 'body', 'text']))
        
        if not main_content:
            logger.debug("No main content container found, using entire document")
            main_content = soup
        
        # Extract paragraphs and headings
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        logger.debug(f"Found {len(paragraphs)} paragraphs and headings")
        
        # Join the text from paragraphs and headings
        text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # If no paragraphs found, get all text
        if not text:
            logger.debug("No paragraph text found, extracting all text")
            text = main_content.get_text()
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
        
        return text
    except Exception as e:
        logger.error(f"Error in extract_main_content: {e}")
        traceback.print_exc()
        # Fallback to getting all text
        return soup.get_text()

def post_process_summary(text):
    """
    Post-process the summary to ensure it forms meaningful sentences.
    - Fix spacing issues
    - Ensure proper sentence endings
    - Remove incomplete sentences
    - Fix capitalization
    """
    try:
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([.,;:!?])', r'\1', text)
        
        # Split into sentences for further processing
        sentences = nltk.sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            # Trim whitespace
            sentence = sentence.strip()
            
            # Skip empty sentences
            if not sentence:
                continue
                
            # Ensure sentence starts with capital letter
            if sentence and sentence[0].isalpha() and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
                
            # Ensure sentence ends with proper punctuation
            if not any(sentence.endswith(p) for p in ['.', '!', '?']):
                sentence = sentence + '.'
                
            processed_sentences.append(sentence)
        
        # Join sentences with proper spacing
        processed_text = ' '.join(processed_sentences)
        
        # Final cleanup
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text.strip()
        
        # Check for coherence - ensure sentences are connected logically
        processed_text = ensure_coherence(processed_text)
        
        return processed_text
    except Exception as e:
        logger.error(f"Error in post-processing summary: {e}")
        traceback.print_exc()
        return text

def ensure_coherence(text):
    """
    Ensure the summary has logical coherence by:
    1. Removing sentences that start with conjunctions without context
    2. Ensuring proper referential integrity (pronouns have antecedents)
    3. Removing sentences that seem disconnected
    """
    try:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return text
            
        coherent_sentences = [sentences[0]]  # Always keep the first sentence
        
        # List of conjunctions and pronouns that might indicate a dependent sentence
        conjunctions = ['and', 'but', 'or', 'yet', 'so', 'for', 'nor', 'because', 'although', 'since', 'unless', 'while']
        pronouns = ['he', 'she', 'it', 'they', 'them', 'their', 'this', 'that', 'these', 'those']
        
        for i in range(1, len(sentences)):
            current = sentences[i]
            words = current.lower().split()
            
            # Skip sentences that start with conjunctions and are short
            if words and words[0] in conjunctions and len(words) < 8:
                continue
                
            # Check if sentence starts with a pronoun without clear antecedent
            if words and words[0] in pronouns and i > 1:
                # Keep it only if the previous sentence likely provides context
                prev_words = sentences[i-1].lower().split()
                if len(prev_words) > 5:  # Previous sentence has enough context
                    coherent_sentences.append(current)
            else:
                coherent_sentences.append(current)
        
        return ' '.join(coherent_sentences)
    except Exception as e:
        logger.error(f"Error ensuring coherence: {e}")
        return text

def generate_audio_google_tts(text, language='en'):
    """Generate audio using Google Translate TTS API with caching."""
    try:
        # Check cache first
        if REDIS_AVAILABLE:
            cache_key = generate_cache_key('audio', {'text': text, 'lang': language})
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Using cached audio for language: {language}")
                return pickle.loads(cached_result)
        
        logger.info(f"Generating audio with Google TTS for language: {language}")
        
        # Map language codes to Google TTS language codes
        language_map = {
            'en': 'en',
            'hi': 'hi',
            'kn': 'kn'
        }
        
        # Use the mapped language or default to English
        tts_lang = language_map.get(language, 'en')
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file_path = temp_file.name
        
        # Construct the Google TTS URL
        base_url = "https://translate.google.com/translate_tts"
        
        # Different chunking strategies based on language
        max_chunk_size = 100 if language in ['hi', 'kn'] else 200
        
        # Language-specific text splitting
        if language == 'en':
            # For English, split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            # Combine short sentences to avoid too many small chunks
            all_chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        all_chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                all_chunks.append(current_chunk.strip())
        else:
            # For Hindi and Kannada, use more careful chunking
            # Split on Devanagari and Kannada punctuation as well as Latin punctuation
            if language == 'hi':
                # Hindi/Devanagari punctuation
                split_pattern = r'(?<=[।॥.!?])\s+'
            elif language == 'kn':
                # Kannada punctuation
                split_pattern = r'(?<=[.!?।॥\u0964\u0965])\s+'
            else:
                split_pattern = r'(?<=[.!?])\s+'
                
            sentences = re.split(split_pattern, text)
            
            # For non-English languages, use smaller chunks and ensure we don't break words
            all_chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        all_chunks.append(current_chunk.strip())
                    
                    # If the sentence itself is too long, break it into smaller pieces
                    if len(sentence) > max_chunk_size:
                        # For Hindi/Kannada, try to break at word boundaries
                        words = sentence.split()
                        sub_chunk = ""
                        for word in words:
                            if len(sub_chunk) + len(word) < max_chunk_size:
                                sub_chunk += " " + word if sub_chunk else word
                            else:
                                if sub_chunk:
                                    all_chunks.append(sub_chunk.strip())
                                sub_chunk = word
                        if sub_chunk:
                            all_chunks.append(sub_chunk.strip())
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                all_chunks.append(current_chunk.strip())
        
        # Ensure we have at least one chunk
        if not all_chunks:
            all_chunks = [text[:max_chunk_size]]
        
        # Remove empty chunks
        all_chunks = [chunk for chunk in all_chunks if chunk.strip()]
        
        original_chunk_count = len(all_chunks)
        logger.debug(f"Original chunk count: {original_chunk_count}")
        
        # For non-English languages, we'll process fewer chunks in parallel and use sequential processing
        # to ensure correct ordering and avoid rate limiting
        if language in ['hi', 'kn']:
            logger.info(f"Using optimized parallel processing for {language} language")
            
            # Function to download a single chunk with timeout - optimized for non-English
            def download_chunk_non_english(chunk_idx, chunk_text):
                params = {
                    'ie': 'UTF-8',
                    'q': chunk_text,
                    'tl': tts_lang,
                    'client': 'tw-ob',
                    'ttsspeed': '0.9' if language in ['hi', 'kn'] else '1.0'  # Slightly slower for better clarity
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://translate.google.com/',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                try:
                    # Use a longer timeout for non-English languages
                    response = requests.get(base_url, params=params, headers=headers, timeout=5.0)
                    response.raise_for_status()
                    
                    # Add a small randomized delay to avoid rate limiting
                    time.sleep(random.uniform(0.1, 0.3))
                    
                    return chunk_idx, response.content
                except Exception as e:
                    logger.error(f"Error downloading chunk {chunk_idx}: {e}")
                    return chunk_idx, None
            
            # Use parallel processing with ThreadPoolExecutor but with fewer workers
            chunk_data = []
            
            # Use fewer workers for non-English to avoid rate limiting
            max_workers = min(3, len(all_chunks))
            
            # Randomize the order of chunks to distribute load
            chunk_indices = list(range(len(all_chunks)))
            random.shuffle(chunk_indices)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks in randomized order
                future_to_chunk = {
                    executor.submit(download_chunk_non_english, i, all_chunks[i]): (i, all_chunks[i]) 
                    for i in chunk_indices
                }
                
                # Set a timeout for all tasks together - longer for non-English
                total_timeout = min(30.0, 3.0 * len(all_chunks))
                done, not_done = concurrent.futures.wait(
                    future_to_chunk.keys(), 
                    timeout=total_timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # Cancel any remaining tasks
                for future in not_done:
                    future.cancel()
                
                # Process completed tasks
                for future in done:
                    chunk_idx, content = future.result()
                    if content:
                        chunk_data.append((chunk_idx, content))
                        logger.info(f"Downloaded chunk {chunk_idx+1}/{len(all_chunks)}")
        
        else:
            # For English, use parallel processing with more workers
            # Function to download a single chunk with timeout
            def download_chunk(chunk_idx, chunk_text):
                params = {
                    'ie': 'UTF-8',
                    'q': chunk_text,
                    'tl': tts_lang,
                    'client': 'tw-ob',
                    'ttsspeed': '1.0'  # Normal speed
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://translate.google.com/',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
                
                try:
                    # Add a strict timeout to avoid hanging requests
                    response = requests.get(base_url, params=params, headers=headers, timeout=3.0)
                    response.raise_for_status()
                    return chunk_idx, response.content
                except Exception as e:
                    logger.error(f"Error downloading chunk {chunk_idx}: {e}")
                    return chunk_idx, None
            
            # Use parallel processing with ThreadPoolExecutor
            chunk_data = []
            
            # Optimize with parallel processing but with a limit to avoid rate limiting
            max_workers = min(4, len(all_chunks))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(download_chunk, i, chunk): (i, chunk) 
                    for i, chunk in enumerate(all_chunks)
                }
                
                # Set a timeout for all tasks together
                total_timeout = min(20.0, 2.0 * len(all_chunks))
                done, not_done = concurrent.futures.wait(
                    future_to_chunk.keys(), 
                    timeout=total_timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # Cancel any remaining tasks
                for future in not_done:
                    future.cancel()
                
                # Process completed tasks
                for future in done:
                    chunk_idx, content = future.result()
                    if content:
                        chunk_data.append((chunk_idx, content))
                        logger.info(f"Downloaded chunk {chunk_idx+1}/{len(all_chunks)}")
        
        # Sort chunks by their original index
        chunk_data.sort(key=lambda x: x[0])
        
        # Combine audio chunks in correct order
        with open(temp_file_path, 'wb') as output_file:
            for _, content in chunk_data:
                output_file.write(content)
        
        # Read the audio file and encode it as base64
        with open(temp_file_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Warning: Failed to delete temporary file {temp_file_path}: {e}")
        
        # Cache the audio data
        if REDIS_AVAILABLE and audio_data:
            # Cache audio for 7 days (604800 seconds)
            redis_client.setex(cache_key, 604800, pickle.dumps(audio_data))
            logger.info(f"Cached audio for language: {language}")
        
        return audio_data
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        traceback.print_exc()
        return None

def clean_text(text):
    """Clean the text by removing citations, extra whitespace, etc."""
    # Remove citations in the format [number] or [number, number, ...]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    return text

def devanagari_braille_conversion(text):
    """
    Convert Hindi text to Devanagari/Bharati Braille.
    This implements the Bharati Braille standard for Hindi.
    """
    try:
        # Braille mapping for Hindi/Devanagari
        # Based on Bharati Braille standard
        devanagari_to_braille = {
            # Vowels
            'अ': '⠁',    # a
            'आ': '⠜',    # aa
            'इ': '⠊',    # i
            'ई': '⠔',    # ii
            'उ': '⠥',    # u
            'ऊ': '⠳',    # uu
            'ऋ': '⠐⠗',   # ri
            'ॠ': '⠐⠗⠗',  # rri
            'ऌ': '⠐⠇',   # li
            'ॡ': '⠐⠇⠇',  # lli
            'ए': '⠑',    # e
            'ऐ': '⠌',    # ai
            'ओ': '⠕',    # o
            'औ': '⠪',    # au
            'अं': '⠁⠰',   # am (anusvara)
            'अः': '⠁⠠',   # aha (visarga)
            
            # Consonants
            'क': '⠅',    # ka
            'ख': '⠨',    # kha
            'ग': '⠛',    # ga
            'घ': '⠣',    # gha
            'ङ': '⠬',    # nga
            'च': '⠉',    # ca
            'छ': '⠡',    # cha
            'ज': '⠚',    # ja
            'झ': '⠴',    # jha
            'ञ': '⠒',    # nya
            'ट': '⠾',    # Ta
            'ठ': '⠺',    # Tha
            'ड': '⠫',    # Da
            'ढ': '⠿',    # Dha
            'ण': '⠼',    # Na
            'त': '⠞',    # ta
            'थ': '⠹',    # tha
            'द': '⠙',    # da
            'ध': '⠮',    # dha
            'न': '⠝',    # na
            'प': '⠏',    # pa
            'फ': '⠖',    # pha
            'ब': '⠃',    # ba
            'भ': '⠘',    # bha
            'म': '⠍',    # ma
            'य': '⠽',    # ya
            'र': '⠗',    # ra
            'ल': '⠇',    # la
            'व': '⠧',    # va
            'श': '⠩',    # sha
            'ष': '⠯',    # Sha
            'स': '⠎',    # sa
            'ह': '⠓',    # ha
            'क्ष': '⠅⠈⠯',  # ksha
            'त्र': '⠞⠈⠗',  # tra
            'ज्ञ': '⠚⠈⠒',  # gya
            'श्र': '⠩⠈⠗',  # shra
            
            # Half consonants (consonant + halant)
            'क्': '⠅⠈',   # k
            'ख्': '⠨⠈',   # kh
            'ग्': '⠛⠈',   # g
            'घ्': '⠣⠈',   # gh
            'ङ्': '⠬⠈',   # ng
            'च्': '⠉⠈',   # ch
            'छ्': '⠡⠈',   # chh
            'ज्': '⠚⠈',   # j
            'झ्': '⠴⠈',   # jh
            'ञ्': '⠒⠈',   # ny
            'ट्': '⠾⠈',   # T
            'ठ्': '⠺⠈',   # Th
            'ड्': '⠫⠈',   # D
            'ढ्': '⠿⠈',   # Dh
            'ण्': '⠼⠈',   # N
            'त्': '⠞⠈',   # t
            'थ्': '⠹⠈',   # th
            'द्': '⠙⠈',   # d
            'ध्': '⠮⠈',   # dh
            'न्': '⠝⠈',   # n
            'प्': '⠏⠈',   # p
            'फ्': '⠖⠈',   # ph
            'ब्': '⠃⠈',   # b
            'भ्': '⠘⠈',   # bh
            'म्': '⠍⠈',   # m
            'य्': '⠽⠈',   # y
            'र्': '⠗⠈',   # r
            'ल्': '⠇⠈',   # l
            'व्': '⠧⠈',   # v
            'श्': '⠩⠈',   # sh
            'ष्': '⠯⠈',   # Sh
            'स्': '⠎⠈',   # s
            'ह्': '⠓⠈',   # h
            
            # Matras (vowel signs)
            'ा': '⠜',    # aa
            'ि': '⠊',    # i
            'ी': '⠔',    # ii
            'ु': '⠥',    # u
            'ू': '⠳',    # uu
            'ृ': '⠐⠗',   # ri
            'ॄ': '⠐⠗⠗',  # rri
            'ॢ': '⠐⠇',   # li
            'ॣ': '⠐⠇⠇',  # lli
            'े': '⠑',    # e
            'ै': '⠌',    # ai
            'ो': '⠕',    # o
            'ौ': '⠪',    # au
            'ं': '⠰',     # anusvara (dot above)
            'ः': '⠠',     # visarga (two dots)
            '्': '⠈',     # halant/virama
            
            # Numerals
            '०': '⠚',    # 0
            '१': '⠁',    # 1
            '२': '⠃',    # 2
            '३': '⠉',    # 3
            '४': '⠙',    # 4
            '५': '⠑',    # 5
            '६': '⠋',    # 6
            '७': '⠛',    # 7
            '८': '⠓',    # 8
            '९': '⠊',    # 9
            
            # Punctuation and special characters
            ' ': ' ',     # space
            ',': '⠂',     # comma
            '.': '⠲',     # period
            ';': '⠆',     # semicolon
            ':': '⠒',     # colon
            '?': '⠦',     # question mark
            '!': '⠖',     # exclamation mark
            "'": '⠄',     # apostrophe
            '"': '⠐⠂',    # quotation mark
            '(': '⠐⠣',    # opening parenthesis
            ')': '⠐⠜',    # closing parenthesis
            '-': '⠤',     # hyphen
            '।': '⠲',     # Hindi period (danda)
        }
        
        # Process the text character by character, using the comprehensive mapping
        result = ""
        i = 0
        while i < len(text):
            # Check for specific multi-character combinations first
            if i < len(text) - 2 and text[i:i+3] in devanagari_to_braille:
                result += devanagari_to_braille[text[i:i+3]]
                i += 3
            elif i < len(text) - 1 and text[i:i+2] in devanagari_to_braille:
                result += devanagari_to_braille[text[i:i+2]]
                i += 2
            else:
                # Single character
                result += devanagari_to_braille.get(text[i], "⠿")
                i += 1
        
        return result
    except Exception as e:
        logger.error(f"Error in Devanagari braille conversion: {e}")
        return "Error in braille conversion"

def kannada_braille_conversion(text):
    """
    Convert Kannada text to Bharati Braille.
    This implements the Bharati Braille standard for Kannada.
    """
    try:
        # Braille mapping for Kannada
        # Based on Bharati Braille standard
        kannada_to_braille = {
            # Vowels
            'ಅ': '⠁',    # a
            'ಆ': '⠜',    # aa
            'ಇ': '⠊',    # i
            'ಈ': '⠔',    # ii
            'ಉ': '⠥',    # u
            'ಊ': '⠳',    # uu
            'ಋ': '⠐⠗',   # r̥
            'ೠ': '⠐⠗⠗',  # r̥̄
            'ಌ': '⠐⠇',   # l̥
            'ೡ': '⠐⠇⠇',  # l̥̄  
            'ಎ': '⠢',    # e
            'ಏ': '⠑',    # ee
            'ಐ': '⠌',    # ai
            'ಒ': '⠭',    # o
            'ಓ': '⠕',    # oo
            'ಔ': '⠪',    # au
            'ಅಂ': '⠁⠰',  # aṃ (anusvara)
            'ಅಃ': '⠁⠠',  # aḥ (visarga)
            
            # Consonants
            'ಕ': '⠅',    # ka
            'ಖ': '⠨',    # kha
            'ಗ': '⠛',    # ga
            'ಘ': '⠣',    # gha
            'ಙ': '⠬',    # ṅa
            'ಚ': '⠉',    # ca
            'ಛ': '⠡',    # cha
            'ಜ': '⠚',    # ja
            'ಝ': '⠴',    # jha
            'ಞ': '⠒',    # ña
            'ಟ': '⠾',    # ṭa
            'ಠ': '⠺',    # ṭha
            'ಡ': '⠫',    # ḍa
            'ಢ': '⠿',    # ḍha
            'ಣ': '⠼',    # ṇa
            'ತ': '⠞',    # ta
            'ಥ': '⠹',    # tha
            'ದ': '⠙',    # da
            'ಧ': '⠮',    # dha
            'ನ': '⠝',    # na
            'ಪ': '⠏',    # pa
            'ಫ': '⠖',    # pha
            'ಬ': '⠃',    # ba
            'ಭ': '⠘',    # bha
            'ಮ': '⠍',    # ma
            'ಯ': '⠽',    # ya
            'ರ': '⠗',    # ra
            'ಱ': '⠱',    # ra (alternate)
            'ಲ': '⠇',    # la
            'ವ': '⠧',    # va
            'ಶ': '⠩',    # śa
            'ಷ': '⠯',    # ṣa
            'ಸ': '⠎',    # sa
            'ಹ': '⠓',    # ha
            'ಳ': '⠸⠇',   # ḷa
            'ೞ': '⠻',    # ḻa (rare)
            
            # Common Conjunct Consonants 
            'ಕ್ಷ': '⠅⠈⠯',  # kṣa
            'ತ್ರ': '⠞⠈⠗',  # tra
            'ಜ್ಞ': '⠚⠈⠒',  # jña
            'ಶ್ರ': '⠩⠈⠗',  # śra
            'ದ್ರ': '⠙⠈⠗',  # dra
            'ಪ್ರ': '⠏⠈⠗',  # pra
            'ನ್ನ': '⠝⠈⠝',  # nna
            'ತ್ತ': '⠞⠈⠞',  # tta
            'ಸ್ಥ': '⠎⠈⠹',  # stha
            'ದ್ದ': '⠙⠈⠙',  # dda
            'ಲ್ಲ': '⠇⠈⠇',  # lla
            'ಗ್ಗ': '⠛⠈⠛',  # gga 
            'ಚ್ಚ': '⠉⠈⠉',  # cca
            'ಬ್ಬ': '⠃⠈⠃',  # bba
            
            # Half consonants (consonant + virama)
            'ಕ್': '⠅⠈',   # k
            'ಖ್': '⠨⠈',   # kh
            'ಗ್': '⠛⠈',   # g
            'ಘ್': '⠣⠈',   # gh
            'ಙ್': '⠬⠈',   # ṅg
            'ಚ್': '⠉⠈',   # c
            'ಛ್': '⠡⠈',   # ch
            'ಜ್': '⠚⠈',   # j
            'ಝ್': '⠴⠈',   # jh
            'ಞ್': '⠒⠈',   # ñ
            'ಟ್': '⠾⠈',   # ṭ
            'ಠ್': '⠺⠈',   # ṭh
            'ಡ್': '⠫⠈',   # ḍ
            'ಢ್': '⠿⠈',   # ḍh
            'ಣ್': '⠼⠈',   # ṇ
            'ತ್': '⠞⠈',   # t
            'ಥ್': '⠹⠈',   # th
            'ದ್': '⠙⠈',   # d
            'ಧ್': '⠮⠈',   # dh
            'ನ್': '⠝⠈',   # n
            'ಪ್': '⠏⠈',   # p
            'ಫ್': '⠖⠈',   # ph
            'ಬ್': '⠃⠈',   # b
            'ಭ್': '⠘⠈',   # bh
            'ಮ್': '⠍⠈',   # m
            'ಯ್': '⠽⠈',   # y
            'ರ್': '⠗⠈',   # r
            'ಱ್': '⠱⠈',   # r (alternate)
            'ಲ್': '⠇⠈',   # l
            'ವ್': '⠧⠈',   # v
            'ಶ್': '⠩⠈',   # ś
            'ಷ್': '⠯⠈',   # ṣ
            'ಸ್': '⠎⠈',   # s
            'ಹ್': '⠓⠈',   # h
            'ಳ್': '⠸⠇⠈', # ḷ
            'ೞ್': '⠻⠈',   # ḻ (rare)
            
            # Matras (vowel signs)
            'ಾ': '⠜',    # aa
            'ಿ': '⠊',    # i
            'ೀ': '⠔',    # ii
            'ು': '⠥',    # u
            'ೂ': '⠳',    # uu
            'ೃ': '⠐⠗',   # r̥
            'ೄ': '⠐⠗⠗',  # r̥̄
            'ೢ': '⠐⠇',   # l̥
            'ೣ': '⠐⠇⠇',  # l̥̄
            'ೆ': '⠢',    # e
            'ೇ': '⠑',    # ee
            'ೈ': '⠌',    # ai
            'ೊ': '⠭',    # o
            'ೋ': '⠕',    # oo
            'ೌ': '⠪',    # au
            'ಂ': '⠰',    # ṃ (anusvara)
            'ಃ': '⠠',    # ḥ (visarga)
            '್': '⠈',    # virama (halant) 
            
            # Numerals
            '೦': '⠚',    # 0
            '೧': '⠁',    # 1
            '೨': '⠃',    # 2
            '೩': '⠉',    # 3
            '೪': '⠙',    # 4
            '೫': '⠑',    # 5
            '೬': '⠋',    # 6
            '೭': '⠛',    # 7
            '೮': '⠓',    # 8
            '೯': '⠊',    # 9
            
            # Punctuation and special characters
            ' ': ' ',     # space
            ',': '⠂',     # comma
            '.': '⠲',     # period
            ';': '⠆',     # semicolon
            ':': '⠒',     # colon
            '?': '⠦',     # question mark
            '!': '⠖',     # exclamation mark
            "'": '⠄',     # apostrophe
            '"': '⠐⠂',    # quotation mark
            '(': '⠐⠣',    # opening parenthesis
            ')': '⠐⠜',    # closing parenthesis
            '-': '⠤',     # hyphen
            '।': '⠲',     # danda (equivalent to period)
            '॥': '⠲⠲',    # double danda
        }
        
        # Process the text character by character, using the comprehensive mapping
        result = ""
        i = 0
        while i < len(text):
            # Check for specific multi-character combinations first
            if i < len(text) - 2 and text[i:i+3] in kannada_to_braille:
                result += kannada_to_braille[text[i:i+3]]
                i += 3
            elif i < len(text) - 1 and text[i:i+2] in kannada_to_braille:
                result += kannada_to_braille[text[i:i+2]]
                i += 2
            else:
                # Single character
                result += kannada_to_braille.get(text[i], "⠿")
                i += 1
        
        return result
    except Exception as e:
        logger.error(f"Error in Kannada braille conversion: {e}")
        return "Error in braille conversion"

def simple_braille_conversion(text):
    """
    A function that converts text to a basic representation of braille.
    Handles letters, numbers, and special characters according to standard braille patterns.
    """
    try:
        # English alphabet mapping (lowercase)
        english_braille_map = {
            'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
            'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
            'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
            'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
            'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
            'z': '⠵'
        }
        
        # Number mapping - in standard braille, numbers use the same patterns as the first 10 letters with a number sign
        number_sign = '⠼'
        number_map = {
            '1': '⠼⠁', '2': '⠼⠃', '3': '⠼⠉', '4': '⠼⠙', '5': '⠼⠑',
            '6': '⠼⠋', '7': '⠼⠛', '8': '⠼⠓', '9': '⠼⠊', '0': '⠼⠚'
        }
        
        # Punctuation and special characters
        punctuation_map = {
            ' ': ' ',  # Space
            ',': '⠂',  # Comma
            '.': '⠲',  # Period
            ';': '⠆',  # Semicolon
            '?': '⠦',  # Question mark
            '!': '⠖',  # Exclamation mark
            '"': '⠐⠂',  # Quotation mark
            "'": '⠄',  # Apostrophe
            '-': '⠤',  # Hyphen
            '/': '⠌',  # Slash
            '\\': '⠸⠌',  # Backslash
            ':': '⠒',  # Colon
            '(': '⠐⠣',  # Opening parenthesis
            ')': '⠐⠜',  # Closing parenthesis
            '[': '⠐⠣',  # Opening bracket (similar to parenthesis)
            ']': '⠐⠜',  # Closing bracket (similar to parenthesis)
            '{': '⠐⠣',  # Opening brace (similar to parenthesis)
            '}': '⠐⠜',  # Closing brace (similar to parenthesis)
            '@': '⠈⠁',  # At sign
            '#': '⠼⠓',  # Hash/number sign
            '$': '⠈⠎',  # Dollar sign
            '%': '⠨⠴',  # Percent sign
            '&': '⠈⠯',  # Ampersand
            '*': '⠐⠔',  # Asterisk
            '+': '⠐⠖',  # Plus sign
            '=': '⠐⠶',  # Equals sign
            '<': '⠐⠅',  # Less than sign
            '>': '⠐⠂',  # Greater than sign
            '_': '⠸⠤',  # Underscore
            '|': '⠸⠇',  # Vertical bar
            '~': '⠐⠢',  # Tilde
            '`': '⠈',   # Backtick
            '…': '⠲⠲⠲', # Ellipsis
            '•': '⠸⠲',  # Bullet point
            '—': '⠤⠤',  # Em dash
            '–': '⠤',   # En dash
            '«': '⠦',   # Left double angle quotation mark
            '»': '⠴',   # Right double angle quotation mark
            '„': '⠰⠂',  # Double low quotation mark
            '"': '⠰⠄',  # Left double quotation mark
            '"': '⠘⠄',  # Right double quotation mark
            ''': '⠠⠄',  # Left single quotation mark
            ''': '⠄',   # Right single quotation mark
            '©': '⠐⠉',  # Copyright
            '®': '⠐⠗',  # Registered trademark
            '™': '⠐⠞⠍', # Trademark
            '°': '⠘⠨',  # Degree
            '′': '⠄',   # Prime
            '″': '⠄⠄',  # Double prime
            '§': '⠠⠎',  # Section
            '¶': '⠠⠏',  # Paragraph
            '№': '⠼⠙',  # Numero sign
            '†': '⠸⠻',  # Dagger
            '‡': '⠸⠸⠻', # Double dagger
            '✓': '⠸⠩',  # Check mark
            '✗': '⠸⠭',  # Cross mark
            '→': '⠱',   # Right arrow
            '←': '⠣',   # Left arrow
            '↑': '⠋',   # Up arrow
            '↓': '⠙',   # Down arrow
            '⇒': '⠀⠱',  # Right double arrow
            '⇐': '⠀⠣',  # Left double arrow
            '≈': '⠀⠐⠶⠀', # Approximately equal
            '≠': '⠀⠐⠶⠌⠀', # Not equal
            '≤': '⠀⠐⠅⠶⠀', # Less than or equal
            '≥': '⠀⠐⠢⠶⠀', # Greater than or equal
            '∞': '⠿',   # Infinity
            '±': '⠐⠖⠤', # Plus-minus
            '×': '⠐⠦',  # Multiplication
            '÷': '⠐⠌',  # Division
            '√': '⠗⠞',  # Square root
            '∑': '⠎⠍',  # Summation (sigma)
            '∏': '⠏⠗',  # Product (pi)
            'π': '⠏⠊',  # Pi
        }
        
        # Handle common combinations like contractions and abbreviations
        word_contractions = {
            # Common contractions
            "can't": '⠉⠁⠝⠠⠞',
            "don't": '⠙⠕⠝⠠⠞',
            "won't": '⠺⠕⠝⠠⠞',
            "isn't": '⠊⠎⠝⠠⠞',
            "doesn't": '⠙⠕⠑⠎⠝⠠⠞',
            "wouldn't": '⠺⠕⠥⠠⠇⠙⠝⠠⠞',
            "couldn't": '⠉⠕⠥⠠⠇⠙⠝⠠⠞',
            "shouldn't": '⠎⠓⠕⠥⠠⠇⠙⠝⠠⠞',
            "hadn't": '⠓⠁⠙⠝⠠⠞',
            "hasn't": '⠓⠁⠎⠝⠠⠞',
            "haven't": '⠓⠁⠧⠑⠝⠠⠞',
            "mustn't": '⠍⠥⠎⠞⠝⠠⠞',
            "I'm": '⠊⠠⠍',
            "I'll": '⠊⠠⠇⠇',
            "I've": '⠊⠠⠧⠑',
            "I'd": '⠊⠠⠙',
            "you're": '⠽⠕⠥⠠⠗⠑',
            "you'll": '⠽⠕⠥⠠⠇⠇',
            "you've": '⠽⠕⠥⠠⠧⠑',
            "you'd": '⠽⠕⠥⠠⠙',
            "he's": '⠓⠑⠠⠎',
            "he'll": '⠓⠑⠠⠇⠇',
            "he'd": '⠓⠑⠠⠙',
            "she's": '⠎⠓⠑⠠⠎',
            "she'll": '⠎⠓⠑⠠⠇⠇',
            "she'd": '⠎⠓⠑⠠⠙',
            "it's": '⠊⠞⠠⠎',
            "it'll": '⠊⠞⠠⠇⠇',
            "it'd": '⠊⠞⠠⠙',
            "we're": '⠺⠑⠠⠗⠑',
            "we'll": '⠺⠑⠠⠇⠇',
            "we've": '⠺⠑⠠⠧⠑',
            "we'd": '⠺⠑⠠⠙',
            "they're": '⠞⠓⠑⠽⠠⠗⠑',
            "they'll": '⠞⠓⠑⠽⠠⠇⠇',
            "they've": '⠞⠓⠑⠽⠠⠧⠑',
            "they'd": '⠞⠓⠑⠽⠠⠙',
            "that's": '⠞⠓⠁⠞⠠⠎',
            "that'll": '⠞⠓⠁⠞⠠⠇⠇',
            "that'd": '⠞⠓⠁⠞⠠⠙',
            "who's": '⠺⠓⠕⠠⠎',
            "who'll": '⠺⠓⠕⠠⠇⠇',
            "who'd": '⠺⠓⠕⠠⠙',
            "what's": '⠺⠓⠁⠞⠠⠎',
            "what'll": '⠺⠓⠁⠞⠠⠇⠇',
            "what'd": '⠺⠓⠁⠞⠠⠙',
            "where's": '⠺⠓⠑⠗⠑⠠⠎',
            "where'll": '⠺⠓⠑⠗⠑⠠⠇⠇',
            "where'd": '⠺⠓⠑⠗⠑⠠⠙',
            "when's": '⠺⠓⠑⠝⠠⠎',
            "when'll": '⠺⠓⠑⠝⠠⠇⠇',
            "when'd": '⠺⠓⠑⠝⠠⠙',
            "why's": '⠓⠕⠺⠠⠎',
            "why'll": '⠓⠕⠺⠠⠇⠇',
            "why'd": '⠓⠕⠺⠠⠙',
            "how's": '⠓⠕⠺⠠⠎',
            "how'll": '⠓⠕⠺⠠⠇⠇',
            "how'd": '⠓⠕⠺⠠⠙',
            # Common abbreviations
            "Mr.": '⠠⠍⠗⠲',
            "Mrs.": '⠠⠍⠗⠎⠲',
            "Ms.": '⠠⠍⠎⠲',
            "Dr.": '⠠⠙⠗⠲',
            "Prof.": '⠠⠏⠗⠕⠋⠲',
            "etc.": '⠑⠞⠉⠲',
            "i.e.": '⠊⠲⠑⠲',
            "e.g.": '⠑⠲⠛⠲',
            "vs.": '⠧⠎⠲',
            "Fig.": '⠠⠋⠊⠛⠲',
            "Dept.": '⠠⠙⠑⠏⠞⠲',
            "Co.": '⠠⠉⠕⠲',
            "Inc.": '⠠⠊⠝⠉⠲',
            "Ltd.": '⠠⠇⠞⠙⠲',
            "St.": '⠠⠎⠞⠲',
            "Rd.": '⠠⠗⠙⠲',
            "Ave.": '⠠⠁⠧⠑⠲',
            "Blvd.": '⠠⠃⠇⠧⠙⠲',
            "Apt.": '⠠⠁⠏⠞⠲',
            "No.": '⠠⠝⠕⠲',
            "Vol.": '⠠⠧⠕⠇⠲',
            "Jan.": '⠠⠚⠁⠝⠲',
            "Feb.": '⠠⠋⠑⠃⠲',
            "Mar.": '⠠⠍⠁⠗⠲',
            "Apr.": '⠠⠁⠏⠗⠲',
            "Jun.": '⠠⠚⠥⠝⠲',
            "Jul.": '⠠⠚⠥⠇⠲',
            "Aug.": '⠠⠁⠥⠛⠲',
            "Sep.": '⠠⠎⠑⠏⠲',
            "Sept.": '⠠⠎⠑⠏⠞⠲',
            "Oct.": '⠠⠕⠉⠞⠲',
            "Nov.": '⠠⠝⠕⠧⠲',
            "Dec.": '⠠⠙⠑⠉⠲',
            "Mon.": '⠠⠍⠕⠝⠲',
            "Tue.": '⠠⠞⠥⠑⠲',
            "Wed.": '⠠⠺⠑⠙⠲',
            "Thu.": '⠠⠞⠓⠥⠲',
            "Fri.": '⠠⠋⠗⠊⠲',
            "Sat.": '⠠⠎⠁⠞⠲',
            "Sun.": '⠠⠎⠥⠝⠲',
        }
        
        # Check for common contractions and abbreviations first
        words = text.split()
        result = []
        
        for word in words:
            if word in word_contractions:
                result.append(word_contractions[word])
            else:
                # Convert each character in the word
                word_result = ""
                for char in word:
                    if char.isalpha():
                        if char.isupper():
                            # Capital sign followed by the letter
                            word_result += '⠠' + english_braille_map.get(char.lower(), "⠿")
                        else:
                            word_result += english_braille_map.get(char, "⠿")
                    elif char.isdigit():
                        word_result += number_map.get(char, "⠿")
                    else:
                        word_result += punctuation_map.get(char, "⠿")
                result.append(word_result)
        
        # Join the words with spaces
        return " ".join(result)
    except Exception as e:
        logger.error(f"Error in English braille conversion: {e}")
        return "Error in braille conversion"

def mixed_language_braille_conversion(text, language):
    """
    Convert mixed-language text to Braille by identifying and handling:
    1. Target language text (Hindi or Kannada)
    2. English words that didn't get translated (proper nouns, etc.)
    3. Punctuation and spaces
    """
    try:
        result = ""
        # Split text into words
        words = text.split()
        
        for word in words:
            # Check if word contains any target language characters
            if language == 'hi' and re.search(r'[\u0900-\u097F]', word):
                # Word contains Devanagari - process with Devanagari converter
                # Split word into segments (Devanagari and non-Devanagari)
                segments = re.findall(r'[\u0900-\u097F]+|[^\u0900-\u097F]+', word)
                for segment in segments:
                    if re.match(r'[\u0900-\u097F]', segment[0]):
                        # Devanagari segment
                        result += devanagari_braille_conversion(segment)
                    else:
                        # Non-Devanagari segment (English, numbers, etc.)
                        result += simple_braille_conversion(segment)
            elif language == 'kn' and re.search(r'[\u0C80-\u0CFF]', word):
                # Word contains Kannada - process with Kannada converter
                # Split word into segments (Kannada and non-Kannada)
                segments = re.findall(r'[\u0C80-\u0CFF]+|[^\u0C80-\u0CFF]+', word)
                for segment in segments:
                    if re.match(r'[\u0C80-\u0CFF]', segment[0]):
                        # Kannada segment
                        result += kannada_braille_conversion(segment)
                    else:
                        # Non-Kannada segment (English, numbers, etc.)
                        result += simple_braille_conversion(segment)
            else:
                # Word is entirely in English or other non-target language
                # Convert using English Braille rules
                result += simple_braille_conversion(word)
            
            # Add space between words
            result += " "
        
        # Remove trailing space
        if result:
            result = result[:-1]
            
        return result
    except Exception as e:
        logger.error(f"Error in mixed language braille conversion: {e}")
        return "Error in braille conversion"

def detect_language(text):
    """
    Detect the primary language of a given text segment.
    Returns: 'en' for English, 'hi' for Hindi, 'kn' for Kannada, 'mixed' for mixed content
    """
    # Character ranges for different scripts
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
    kannada_chars = re.findall(r'[\u0C80-\u0CFF]', text)
    latin_chars = re.findall(r'[a-zA-Z]', text)
    
    # Count characters from each script
    devanagari_count = len(devanagari_chars)
    kannada_count = len(kannada_chars)
    latin_count = len(latin_chars)
    
    # Determine primary language based on character frequency
    total_chars = devanagari_count + kannada_count + latin_count
    if total_chars == 0:
        return 'en'  # Default to English for punctuation/numbers only
    
    # Calculate percentages
    devanagari_percent = devanagari_count / total_chars * 100
    kannada_percent = kannada_count / total_chars * 100
    latin_percent = latin_count / total_chars * 100
    
    # Assign language based on dominant script (>60%)
    if devanagari_percent > 60:
        return 'hi'
    elif kannada_percent > 60:
        return 'kn'
    elif latin_percent > 60:
        return 'en'
    else:
        return 'mixed'

def process_mixed_language_summary(text):
    """
    Process a summary that may contain mixed languages (English, Hindi, Kannada).
    Adds appropriate language indicators and formats the text for better readability.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        lang = detect_language(sentence)
        
        # Format based on detected language
        if lang == 'hi':
            # Add Hindi indicator (only if not preceded by another Hindi sentence)
            if len(processed_sentences) == 0 or detect_language(processed_sentences[-1]) != 'hi':
                processed_sentences.append(f"[Hindi] {sentence}")
            else:
                processed_sentences.append(sentence)
        elif lang == 'kn':
            # Add Kannada indicator
            if len(processed_sentences) == 0 or detect_language(processed_sentences[-1]) != 'kn':
                processed_sentences.append(f"[Kannada] {sentence}")
            else:
                processed_sentences.append(sentence)
        elif lang == 'mixed':
            # For mixed sentences, try to translate Hindi/Kannada parts to English
            # This is complex and may require additional processing
            # For now, we'll just mark it as mixed
            processed_sentences.append(f"[Mixed] {sentence}")
        else:
            # English - no special marking needed
            processed_sentences.append(sentence)
    
    # Join sentences back together
    return ' '.join(processed_sentences)

@app.route('/correct-voice-command', methods=['POST'])
def correct_voice_command():
    """
    Correct voice commands using OpenAI GPT-4 with fuzzy matching as fallback.
    
    This endpoint takes a transcribed voice command and:
    1. Attempts to correct it using GPT-4
    2. Falls back to fuzzy matching if GPT-4 is unavailable or fails
    3. Returns both the original and corrected commands
    """
    data = request.json
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided'}), 400
    
    original_command = data['command']
    confidence = data.get('confidence', 0.0)
    
    # Initialize variables
    corrected_command = original_command
    correction_method = 'none'
    correction_confidence = confidence
    
    # Define known command patterns for fuzzy matching
    known_commands = {
        'audio': [
            'play audio', 'play the audio', 'start audio', 'start the audio',
            'pause audio', 'pause the audio', 'stop audio', 'stop the audio',
            'resume audio', 'resume the audio', 'continue audio', 'continue the audio',
            'stop completely', 'stop playback completely', 'terminate audio', 'terminate playback'
        ],
        'language': [
            'switch to english', 'change to english', 'use english',
            'switch to hindi', 'change to hindi', 'use hindi',
            'switch to kannada', 'change to kannada', 'use kannada'
        ],
        'search': [
            'search for', 'look up', 'find information about',
            'search', 'find', 'look for'
        ]
    }
    
    # Common scientific and technical terms that might be misheard
    scientific_terms = {
        # Biology
        'zaikot': 'zygote',
        'zygot': 'zygote',
        'mitoses': 'mitosis',
        'myosis': 'meiosis',
        'dna': 'DNA',
        'rna': 'RNA',
        'genone': 'genome',
        'jeanome': 'genome',
        'kromosome': 'chromosome',
        'chromasome': 'chromosome',
        'nucleas': 'nucleus',
        'nuclius': 'nucleus',
        'sytoplasm': 'cytoplasm',
        'sitoplasm': 'cytoplasm',
        'celwall': 'cell wall',
        'clorophil': 'chlorophyll',
        'clorofill': 'chlorophyll',
        'fotosynthesis': 'photosynthesis',
        'fotosinthesis': 'photosynthesis',
        'bactria': 'bacteria',
        'backteria': 'bacteria',
        'viruss': 'virus',
        'viris': 'virus',
        'protene': 'protein',
        'protien': 'protein',
        'ameno acid': 'amino acid',
        'ameeno acid': 'amino acid',
        'lipid': 'lipid',
        'lippit': 'lipid',
        'carbohydrat': 'carbohydrate',
        'carbohidrate': 'carbohydrate',
        
        # Physics
        'quantam': 'quantum',
        'quantom': 'quantum',
        'nucular': 'nuclear',
        'nuclar': 'nuclear',
        'fisics': 'physics',
        'fysics': 'physics',
        'thermodynamix': 'thermodynamics',
        'thermodynamicks': 'thermodynamics',
        'electromagnetism': 'electromagnetism',
        'electromagnatism': 'electromagnetism',
        'relativity': 'relativity',
        'relativety': 'relativity',
        'gravety': 'gravity',
        'gravitee': 'gravity',
        
        # Chemistry
        'kemistry': 'chemistry',
        'chemistree': 'chemistry',
        'atom': 'atom',
        'attom': 'atom',
        'molecool': 'molecule',
        'mollecule': 'molecule',
        'elament': 'element',
        'elliment': 'element',
        'periodic table': 'periodic table',
        'pereodic table': 'periodic table',
        'reaction': 'reaction',
        'reacshun': 'reaction',
        'oxydation': 'oxidation',
        'oxydization': 'oxidation',
        'acidic': 'acidic',
        'asidic': 'acidic',
        'alkalyne': 'alkaline',
        'alkeline': 'alkaline',
        
        # Mathematics
        'mathematic': 'mathematics',
        'mathmatics': 'mathematics',
        'algebra': 'algebra',
        'algibra': 'algebra',
        'calculas': 'calculus',
        'calcules': 'calculus',
        'geometry': 'geometry',
        'geometree': 'geometry',
        'trignometry': 'trigonometry',
        'trigonometree': 'trigonometry',
        'statistix': 'statistics',
        'statistic': 'statistics',
        'probabilty': 'probability',
        'probablity': 'probability',
        
        # Computer Science
        'algorythm': 'algorithm',
        'algorithem': 'algorithm',
        'programing': 'programming',
        'programmin': 'programming',
        'databas': 'database',
        'datbase': 'database',
        'artificial inteligence': 'artificial intelligence',
        'artifical intelligence': 'artificial intelligence',
        'mashin learning': 'machine learning',
        'machine lernin': 'machine learning',
        'neural network': 'neural network',
        'nural network': 'neural network',
        'deep lernin': 'deep learning',
        'deep learnin': 'deep learning'
    }
    
    # Check for direct matches in scientific terms dictionary first
    if original_command.lower() in scientific_terms:
        corrected_command = scientific_terms[original_command.lower()]
        correction_method = 'dictionary'
        correction_confidence = 0.98
        logger.info(f"Scientific term corrected: '{original_command}' → '{corrected_command}'")
        
        # Prepare the response for dictionary-based correction
        result = {
            'original_command': original_command,
            'corrected_command': corrected_command,
            'correction_method': correction_method,
            'confidence': correction_confidence
        }
        return jsonify(result)
    
    # Try GPT-4 correction if API key is available
    if OPENAI_API_KEY:
        try:
            # Cache key for this correction request
            cache_key = generate_cache_key('command_correction', {
                'command': original_command,
                'confidence': confidence
            })
            
            # Check cache first
            if REDIS_AVAILABLE:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
            
            # Prepare the prompt for GPT-4 with improved context for scientific terms
            system_prompt = """You are an AI assistant that corrects misheard voice commands. 
            Your task is to identify the intended command from potentially misheard text.
            
            Known command categories:
            1. Audio commands: play audio, pause audio, resume audio, stop completely
            2. Language commands: switch to english, switch to hindi, switch to kannada
            3. Search commands: search for [topic], find information about [topic]
            
            For search commands, pay special attention to scientific and technical terms that might be misheard.
            Common examples include:
            - "zaikot" should be corrected to "zygote"
            - "nucular" should be corrected to "nuclear"
            - "fotosynthesis" should be corrected to "photosynthesis"
            - "kromosome" should be corrected to "chromosome"
            - "algorythm" should be corrected to "algorithm"
            
            Respond ONLY with the corrected command. If you're unsure or the command seems unrelated to the known categories, return the original command unchanged."""
            
            user_prompt = f"Correct this potentially misheard voice command: '{original_command}'"
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use GPT-4 for best accuracy
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.3  # Low temperature for more predictable outputs
            )
            
            # Extract the corrected command from the response
            gpt_corrected_command = response.choices[0].message.content.strip()
            
            # Only use GPT's correction if it's different from the original
            if gpt_corrected_command != original_command:
                corrected_command = gpt_corrected_command
                correction_method = 'gpt-4'
                correction_confidence = 0.95  # High confidence for GPT-4 corrections
                logger.info(f"GPT-4 corrected: '{original_command}' → '{corrected_command}'")
            
        except Exception as e:
            logger.error(f"GPT-4 correction failed: {e}")
            # Will fall back to fuzzy matching
    
    # Fuzzy matching fallback
    if correction_method == 'none' or correction_confidence < 0.8:
        try:
            # Import the calculateSimilarity function's logic from JavaScript
            def levenshtein_distance(s1, s2):
                if len(s1) == 0: return len(s2)
                if len(s2) == 0: return len(s1)
                
                matrix = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
                
                for i in range(len(s1) + 1):
                    matrix[i][0] = i
                for j in range(len(s2) + 1):
                    matrix[0][j] = j
                
                for i in range(1, len(s1) + 1):
                    for j in range(1, len(s2) + 1):
                        cost = 0 if s1[i-1] == s2[j-1] else 1
                        matrix[i][j] = min(
                            matrix[i-1][j] + 1,      # deletion
                            matrix[i][j-1] + 1,      # insertion
                            matrix[i-1][j-1] + cost  # substitution
                        )
                
                return matrix[len(s1)][len(s2)]
            
            def calculate_similarity(str1, str2):
                max_length = max(len(str1), len(str2))
                if max_length == 0:
                    return 100  # Both strings are empty
                distance = levenshtein_distance(str1.lower(), str2.lower())
                return round((1 - distance / max_length) * 100)
            
            # Combine known commands with scientific terms for fuzzy matching
            all_known_commands = []
            for category, commands in known_commands.items():
                all_known_commands.extend(commands)
            
            # Add scientific terms to the fuzzy matching pool
            for misspelled, correct in scientific_terms.items():
                all_known_commands.append(correct)
            
            best_match = None
            best_similarity = 0
            
            # Check for exact matches in command parts first
            command_parts = original_command.lower().split()
            for known_command in all_known_commands:
                known_parts = known_command.lower().split()
                
                # Check if any part of the command matches exactly with known commands
                for part in command_parts:
                    if part in known_parts and len(part) > 3:  # Only consider meaningful parts
                        similarity = calculate_similarity(original_command, known_command)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = known_command
            
            # If no good match found by parts, try whole command matching
            if best_similarity < 70:
                for known_command in all_known_commands:
                    similarity = calculate_similarity(original_command, known_command)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = known_command
            
            # Use fuzzy match if it's good enough
            if best_match and best_similarity >= 70:
                corrected_command = best_match
                correction_method = 'fuzzy'
                correction_confidence = best_similarity / 100
                logger.info(f"Fuzzy matching corrected: '{original_command}' → '{corrected_command}' (Similarity: {best_similarity}%)")
        
        except Exception as e:
            logger.error(f"Fuzzy matching fallback failed: {e}")
    
    # Prepare the response
    result = {
        'original_command': original_command,
        'corrected_command': corrected_command,
        'correction_method': correction_method,
        'confidence': correction_confidence
    }
    
    # Cache the result
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(
                cache_key,
                60 * 60,  # Cache for 1 hour
                pickle.dumps(result)
            )
        except Exception as e:
            logger.error(f"Error caching command correction: {e}")
    
    return jsonify(result)

if __name__ == '__main__':
    # Start the cache maintenance thread before running the app
    start_cache_maintenance()
    app.run(debug=True)
