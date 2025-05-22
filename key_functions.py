"""
Missing key functions for the summarization pipeline.
These functions will be imported into app.py to restore full functionality.
"""

import nltk
import re
import traceback
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def summarize_text(url, text_content, post_process_summary_func=None):
    """Summarize the content of a URL using LSA summarizer with fallback methods."""
    try:
        # First try using Sumy's HtmlParser
        try:
            print("Attempting summarization with HtmlParser...")
            parser = HtmlParser.from_url(url, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            summary = []
            for sentence in summarizer(parser.document, 8):
                summary.append(str(sentence))
            
            summary_text = " ".join(summary)
            print(f"HtmlParser summary length: {len(summary_text)} characters")
            
            # If summary is too short, try the fallback method
            if len(summary_text.split()) < 50:
                print("Summary too short, using fallback method")
                raise Exception("Summary too short, using fallback method")
            
            # Post-process the summary to ensure coherence
            if post_process_summary_func:
                summary_text = post_process_summary_func(summary_text)
            return summary_text
            
        except Exception as e:
            print(f"HtmlParser failed: {e}, trying fallback method")
            
            # Fallback: Use PlaintextParser with the extracted text content
            print("Attempting summarization with PlaintextParser...")
            if not text_content or len(text_content.strip()) < 100:
                print("Text content too short for PlaintextParser")
                raise Exception("Text content too short")
                
            parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            summary = []
            for sentence in summarizer(parser.document, 8):
                summary.append(str(sentence))
            
            summary_text = " ".join(summary)
            print(f"PlaintextParser summary length: {len(summary_text)} characters")
            
            # If still too short, use a simple extraction method
            if len(summary_text.split()) < 50:
                print("PlaintextParser summary too short, using extract_key_sentences")
                return extract_key_sentences(text_content, 8, post_process_summary_func)
            
            # Post-process the summary to ensure coherence
            if post_process_summary_func:
                summary_text = post_process_summary_func(summary_text)
            return summary_text
            
    except Exception as e:
        print(f"Error in summarization: {e}")
        traceback.print_exc()
        # Last resort: extract first few sentences
        print("Using extract_key_sentences as last resort")
        return extract_key_sentences(text_content, 8, post_process_summary_func)

def extract_key_sentences(text, count=8, post_process_summary_func=None):
    """Extract key sentences from text as a fallback summarization method."""
    try:
        print("Extracting key sentences...")
        # Tokenize text into sentences
        sentences = nltk.sent_tokenize(text)
        print(f"Found {len(sentences)} sentences")
        
        # Clean sentences and remove very short ones
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            # Remove sentences that are too short or don't end with proper punctuation
            if len(s) > 30:
                # Clean up extra spaces
                s = re.sub(r'\s+', ' ', s)
                cleaned_sentences.append(s)
        
        print(f"After filtering, {len(cleaned_sentences)} sentences remain")
        
        if not cleaned_sentences:
            # If no sentences passed the filter, fall back to the original sentences
            cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
        # Calculate sentence importance based on word frequency
        word_frequencies = {}
        
        # Tokenize all sentences into words and count frequencies
        for sentence in cleaned_sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word.isalnum():  # Only count alphanumeric words
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # Calculate sentence scores based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(cleaned_sentences):
            score = 0
            for word in nltk.word_tokenize(sentence.lower()):
                if word.isalnum() and word in word_frequencies:
                    score += word_frequencies[word]
            sentence_scores[i] = score / max(1, len(nltk.word_tokenize(sentence)))
        
        # Select top sentences by score
        if len(cleaned_sentences) > count:
            # Get indices of top scoring sentences
            top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:count]
            # Sort indices to maintain original order
            top_indices.sort()
            selected_sentences = [cleaned_sentences[i] for i in top_indices]
            print(f"Selected {len(selected_sentences)} top-scoring sentences")
        else:
            # If not enough sentences, take all of them
            selected_sentences = cleaned_sentences
            print(f"Not enough sentences, using all {len(selected_sentences)} available")
        
        # Join the selected sentences with proper spacing
        summary = " ".join(selected_sentences)
        
        # Post-process the summary to ensure coherence
        if post_process_summary_func:
            summary = post_process_summary_func(summary)
        
        print(f"Key sentences summary length: {len(summary)} characters")
        
        # If summary is still too short, return the original text truncated
        if len(summary.split()) < 50 and len(text.split()) > 50:
            print("Summary still too short, truncating original text")
            words = text.split()
            if post_process_summary_func:
                return post_process_summary_func(" ".join(words[:300]) + "...")
            return " ".join(words[:300]) + "..."
            
        return summary
        
    except Exception as e:
        print(f"Error in extracting key sentences: {e}")
        traceback.print_exc()
        # If all else fails, return a portion of the original text
        print("Fallback to truncated original text")
        words = text.split()
        if len(words) > 200:
            if post_process_summary_func:
                return post_process_summary_func(" ".join(words[:300]) + "...")
            return " ".join(words[:300]) + "..."
        return text
