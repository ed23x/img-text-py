import argparse
import base64
import mimetypes
import os
import requests
import sys

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Extract text from an image using Groq API')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()

    image_path = args.image_path
    api_key = os.environ.get('GROQ_API_KEY')

    if not api_key:
        print("GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        return

    try:
        # Read and encode the image
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Determine MIME type
        mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'
        data_url = f"data:{mime_type};base64,{base64_image}"

        # Vision model request
        message_for_vision = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all the text from this image."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }

        request_body_vision = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [message_for_vision],
            "temperature": 0,
            "max_tokens": 7000
        }

        response_vision = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=request_body_vision
        )

        if not response_vision.ok:
            print(f"Vision model request failed with status: {response_vision.status_code}, details: {response_vision.text}", file=sys.stderr)
            return

        data_vision = response_vision.json()
        extracted_text = data_vision['choices'][0]['message']['content']

        # Text model request
        message_for_qwen = {
            "role": "user",
            "content": extracted_text
        }

        request_body_qwen = {
            "model": "qwen-qwq-32b",
            "messages": [message_for_qwen],
            "temperature": 0,
            "max_tokens": 6000
        }

        response_qwen = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=request_body_qwen
        )

        if not response_qwen.ok:
            print(f"Text model request failed with status: {response_qwen.status_code}, details: {response_qwen.text}", file=sys.stderr)
            return

        data_qwen = response_qwen.json()
        final_answer = data_qwen['choices'][0]['message']['content']

        print(final_answer)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
