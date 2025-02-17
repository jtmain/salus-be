from openai import OpenAI
import base64
import os

class AskTheDoc:
    def __init__(self):
        self.key = os.environ["API_KEY"]
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key="key")  

    def ask(self, user_message, lesion_count, image_path):
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        print(f"Sending to OpenAI, User Text: {user_message}, Lesion Count: {lesion_count}, Image Path: {image_path}")

        response = self.client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a skincare and dermatology specialist."},
                {"role": "user", "content": f"User details: {user_message}. Detected {lesion_count} lesions/pimples."},
                {"role": "user", "content": f"Here is the annotated image of the user in Base64 format: {base64_image}"}
            ],
            temperature=0.7
        )

        GPT_response = response.choices[0].message.content
        return GPT_response  