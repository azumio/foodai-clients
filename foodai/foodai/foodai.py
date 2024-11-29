import requests
import json
from PIL import Image
import os
from urllib.parse import urlencode
import io
from foodai.food_response import FoodResponse

class FoodAI:
    """
    A client for the FoodAI API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api-2445582032290.production.gw.apicast.io/v1/foodrecognition"
    
    def analyze(self, image: Image, **kwargs) -> FoodResponse:
        """
        Analyze an image using the FoodAI API.
        :param image: PIL image
        :param kwargs: Additional parameters (e.g. top=5)
        """
        query_params = {"user_key": self.api_key}
        query_params.update(kwargs)  # Add additional parameters

        # Encode query string
        url = f"{self.endpoint}?{urlencode(query_params)}"

        if image.size[0] > 544 or image.size[1] > 544:
            raise ValueError("Image size must be less than or equal to 544x544")

        # Prepare files for the request
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {
            'media': ('img.jpg', img_byte_arr, 'image/jpeg')
        }

        headers = {
            "Accept": "application/json"
        }
            
        response = requests.post(url, headers=headers, files=files)
        if response.ok:
            food_response = FoodResponse.from_dict(response.json())
            return food_response
        raise ValueError(f"Failed to analyze image: {response.text}")

# Direct interface
__client = None

def analyze(image: Image, **kwargs) -> FoodResponse:
    """
    Analyze an image using the globally initialized client.
    :param image: PIL image
    :return: FoodResponse
    """
    api_key = os.environ.get("FOODAI_API_KEY")
    if api_key is None:
        raise ValueError("FOODAI_API_KEY environment variable is not set.")

    global __client
    __client = FoodAI(api_key)
    return __client.analyze(image, **kwargs)