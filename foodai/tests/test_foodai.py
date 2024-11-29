import os
import unittest
from dotenv import load_dotenv
from PIL import Image
from foodai import FoodAI

load_dotenv()

class TestFoodAI(unittest.TestCase):
    def setUp(self):
        # Retrieve API key from environment
        self.api_key = os.getenv("FOODAI_API_KEY")
        if not self.api_key:
            raise ValueError("FOODAI_API_KEY environment variable not set in .env file.")
        self.client = FoodAI(self.api_key)

    def test_analyze(self):
        # Load image
        image = Image.open("tests/fruit_salad.jpg")
        response = self.client.analyze(image)

        self.assertGreater(len(response.results[0]), 0, "Number of results should be greater than 0.")
    
    def test_analyze_top_n(self):
        # Load image
        image = Image.open("tests/fruit_salad.jpg")
        response = self.client.analyze(image, top=5)

        self.assertGreater(len(response.results[0].items), 0, "Number of items should be greated than 0")
        self.assertLessEqual(len(response.results[0].items), 5, "Number of items should be less than or equal to 5")

    
if __name__ == '__main__':
    unittest.main()