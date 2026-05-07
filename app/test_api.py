from openai import OpenAI
from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL

def test_api():
    client = OpenAI(api_key=CAMPUSAI_API_KEY, base_url=CAMPUSAI_URL)
    prompt = "What is the capital of France?"
    response = client.chat.completions.create(
        model="Gemma 4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print("Response:", response.choices[0].message.content)
    assert response.choices[0].message.content, "API call failed or returned no content"

if __name__ == "__main__":
    test_api()