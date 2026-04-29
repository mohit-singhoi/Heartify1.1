# test_gemini.py
# Test script for Google Gemini API using new google.genai package

import sys
import os

print("=" * 60)
print("Google Gemini API Test (New SDK)")
print("=" * 60)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print("=" * 60)

# Step 1: Install the new package if needed
print("\n[1] Checking Google GenAI package...")
try:
    import google.genai as genai
    print("✅ google.genai package is installed (new SDK)")
except ImportError:
    print("❌ google.genai package is NOT installed")
    print("\n📦 To install the new SDK, run:")
    print("   pip install google-genai")
    print("   or")
    print("   C:/Users/mohit/AppData/Local/Programs/Python/Python312/python.exe -m pip install google-genai")
    sys.exit(1)

# Step 2: Check for API key
print("\n[2] Checking for Gemini API key...")

api_key = None

# Method 1: Check Streamlit secrets
try:
    import streamlit as st
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if api_key:
            print("✅ Found API key in Streamlit secrets (.streamlit/secrets.toml)")
        else:
            print("⚠️ No GEMINI_API_KEY found in Streamlit secrets")
    except Exception as e:
        print(f"⚠️ Could not read Streamlit secrets: {e}")
except ImportError:
    print("⚠️ Streamlit not installed (optional for testing)")

# Method 2: Check environment variable
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", None)
    if api_key:
        print("✅ Found API key in environment variable (GEMINI_API_KEY)")
    else:
        print("⚠️ No GEMINI_API_KEY found in environment variables")

if not api_key:
    print("\n❌ NO API KEY FOUND!")
    print("\n📝 To get a free Gemini API key:")
    print("   1. Go to: https://makersuite.google.com/app/apikey")
    print("   2. Sign in with your Google account")
    print("   3. Click 'Create API Key'")
    print("   4. Copy the key")
    sys.exit(1)

# Step 3: Test with new available models
print("\n[3] Testing Gemini API with available models...")

# List of models to try (new model names)
models_to_try = [
    "gemini-2.0-flash",      # Latest fast model
    "gemini-2.0-flash-lite", # Lightweight version
    "gemini-1.5-flash",      # Previous version
    "gemini-1.5-pro",        # Previous pro version
]

success = False
working_model = None

for model_name in models_to_try:
    print(f"\n   Trying model: {model_name}...")
    try:
        # Initialize client with new SDK
        client = genai.Client(api_key=api_key)
        
        # Test with simple prompt
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with only one word: SUCCESS"
        )
        
        print(f"   ✅ Model {model_name} is working!")
        print(f"   Response: {response.text}")
        success = True
        working_model = model_name
        break
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg:
            print(f"   ❌ Model {model_name} not available")
        elif "quota" in error_msg.lower():
            print(f"   ❌ Quota exceeded for {model_name}")
            print(f"   Error: {error_msg[:100]}")
        else:
            print(f"   ❌ Error with {model_name}: {error_msg[:100]}")

if not success:
    print("\n❌ No working model found!")
    print("\n💡 Troubleshooting:")
    print("   1. Check your API key is valid")
    print("   2. Check your internet connection")
    print("   3. Try getting a new API key from https://makersuite.google.com/app/apikey")
    sys.exit(1)

# Step 4: Test with a health-related question
print("\n" + "=" * 60)
print("[4] Testing with a health-related question...")
print("=" * 60)

try:
    client = genai.Client(api_key=api_key)
    
    health_prompt = """You are a helpful cardiovascular health assistant. 
    Answer briefly: What are ideal blood pressure levels?
    Keep response to 2-3 sentences.
    Include a medical disclaimer."""
    
    response = client.models.generate_content(
        model=working_model,
        contents=health_prompt
    )
    
    print("\n📊 AI Response:")
    print("-" * 40)
    print(response.text)
    print("-" * 40)
    print("✅ Health question test PASSED!")
    
except Exception as e:
    print(f"⚠️ Health question test failed: {e}")

# Step 5: Test streaming capability (optional)
print("\n[5] Testing streaming capability...")
try:
    client = genai.Client(api_key=api_key)
    
    stream_response = client.models.generate_content_stream(
        model=working_model,
        contents="List 3 heart health tips. Keep it very brief."
    )
    
    print("✅ Streaming works! Receiving response:")
    for chunk in stream_response:
        if chunk.text:
            print(f"   📝 {chunk.text}", end="", flush=True)
    print("\n")
    
except Exception as e:
    print(f"⚠️ Streaming test skipped: {e}")

# Final summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"✅ Working model: {working_model}")
print("✅ API connection successful")
print("\n🎉 Your Gemini API is ready to use in the Heartify app!")
print("\n🚀 To run the Heartify app:")
print("   streamlit run app.py")
print("=" * 60)