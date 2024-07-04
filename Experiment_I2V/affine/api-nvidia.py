import requests
import base64
from PIL import Image
from io import BytesIO
import requests

url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

payload = {
    "height": 512,
    "width": 512,
    "text_prompts": [
        {
            "text": "A photo of a lady on bike",
            "weight": 1
        }
    ],
    "cfg_scale": 0,
    "clip_guidance_preset": "NONE",
    "sampler": "K_EULER_ANCESTRAL",
    "samples": 1,
    "seed": 0,
    "steps": 4,
    "style_preset": "none"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer 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"
}

response = requests.post(url, json=payload, headers=headers)


# Check if the request was successful
if response.status_code == 200:
    # Extract the image data from the response
    response_data = response.json()
    if 'artifacts' in response_data and response_data['artifacts']:
        artifact = response_data['artifacts'][0]
        image_data_base64 = artifact.get('base64')
        
        if image_data_base64:
            # Decode the base64 string and save the image as a PNG file
            image_data = base64.b64decode(image_data_base64)
            image = Image.open(BytesIO(image_data))
            image.save("shiba_inu_bike.png")
            print("Image saved as shiba_inu_bike.png")
        else:
            print("Image data is empty.")
    else:
        print("No artifacts found in the response.")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)

