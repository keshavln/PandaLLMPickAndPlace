import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT_PATH = 'system_prompt.txt'

groq_client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def load_system_prompt(filepath):
    """
    Loads the system prompt to be given to the LLM.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_object_and_target(user_input):
    """
    Extracts the object and target from the user's prompt.
    """
    system_prompt = load_system_prompt('system_prompt.txt')

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=100,
        )
        
        object_name = chat_completion.choices[0].message.content.strip()
        return object_name
    
    except Exception as e:
        print(f"Error with Groq API: {e}")