#!/usr/bin/env python3
"""Quick test to verify Google Gemini API key is working."""

import os
import sys

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai not installed")
    print("Run: pip install google-genai")
    sys.exit(1)

# Get API key
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not set")
    print("Run: export GOOGLE_API_KEY='your-key-here'")
    sys.exit(1)

print(f"API Key found: {api_key[:10]}...{api_key[-4:]} ({len(api_key)} chars)")

print("Testing Gemini 3 Flash...")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3-flash",
        contents="Say 'API key works!' in exactly 3 words."
    )
    print(f"Response: {response.text}")
    print("\nSUCCESS: API key is valid and working!")
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nPossible issues:")
    print("  - API key is invalid or truncated")
    print("  - API key doesn't have Gemini API access")
    print("  - Quota exceeded")
    sys.exit(1)
