import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "You are a stock trading assistant. Given that AAPL is trading at $150 with RSI of 45, should I buy, sell or hold? Reply in one sentence."
        }
    ],
    max_tokens=100
)

print("Groq connection successful!")
print("Response:", response.choices[0].message.content)