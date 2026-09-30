import re
import time
from typing import List, Dict, Optional

# A pre-compiled fast dictionary of known entities for the hackathon scenario
# We use sets for O(1) lookups and compiled regex for sub-millisecond scanning
KNOWN_CITIES = {
    "boston", "bos", "new york", "nyc", "chicago", "denver", 
    "seattle", "miami", "austin", "san francisco", "sfo", "la", "los angeles"
}

KNOWN_DEVICES = {
    "qn90", "s24", "wf45", "tv", "laptop", "phone", "washer"
}

# Compile high-speed regex to strip punctuation and normalize text
_WORD_PATTERN = re.compile(r'\b\w+\b')

def extract_entities_fast(streaming_text_chunk: str) -> Dict[str, List[str]]:
    """
    Sub-100ms Speculative Entity Extraction.
    Runs on every incoming word chunk BEFORE the user finishes speaking.
    If it hits an entity, it fires an async pre-fetch event to ChromaDB.
    """
    start_time = time.perf_counter()
    
    words = _WORD_PATTERN.findall(streaming_text_chunk.lower())
    
    found_cities = []
    found_devices = []
    
    # Fast O(N) scan over the chunk
    for word in words:
        if word in KNOWN_CITIES:
            found_cities.append(word)
        if word in KNOWN_DEVICES:
            found_devices.append(word)
            
    # Also check multi-word entities
    lower_chunk = streaming_text_chunk.lower()
    for multi_word_city in ["new york", "san francisco", "los angeles"]:
        if multi_word_city in lower_chunk and multi_word_city not in found_cities:
            found_cities.append(multi_word_city)
            
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "cities": found_cities,
        "devices": found_devices,
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    # Simulate a streaming utterance arriving in chunks
    stream = [
        "I need a flight",
        "to",
        "Chicago",
        "wait no",
        "make that",
        "Denver",
        "and look up",
        "the QN90 manual"
    ]
    
    accumulated_text = ""
    for chunk in stream:
        accumulated_text += chunk + " "
        result = extract_entities_fast(accumulated_text)
        
        # We only care about printing if we detected something
        if result["cities"] or result["devices"]:
            print(f"Stream: '{accumulated_text.strip()}'")
            print(f"  -> Detected Entities: Cities={result['cities']}, Devices={result['devices']}")
            print(f"  -> Latency: {result['latency_ms']:.4f} ms\n")
