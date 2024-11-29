from dataclasses import dataclass
from typing import List, Optional
import dacite
import json

@dataclass
class ServingSize:
    unit: str   
    servingWeight: Optional[float] = None # weight in KG
    servingsPerContainer: Optional[float] = None

@dataclass
class Nutrition:
    totalCarbs: Optional[float] = None
    totalFat: Optional[float] = None
    protein: Optional[float] = None
    calories: Optional[float] = None

@dataclass
class Item:
    servingSizes: List[ServingSize]
    nutrition: Nutrition
    name: str
    food_id: str
    score: Optional[int] = None
    group: Optional[str] = None

@dataclass
class Result:
    items: List[Item]
    group: Optional[str] = None

@dataclass
class Timing:
    foodai_totaltime: float
    foodai_classificationtime: float
    proxy_foodairequesttime: float

@dataclass
class FoodResponse:

    @classmethod
    def from_dict(cls, data):
        if "top_results" in data:
            data["results"] = [{
                "items": data.pop("top_results"), 
            }]

        return dacite.from_dict(cls, data,  config=dacite.Config(type_hooks={
                Timing: lambda x: Timing(**x),
                Result: lambda x: Result(**x),
                Item: lambda x: Item(**x),
                ServingSize: lambda x: ServingSize(**x),
                Nutrition: lambda x: Nutrition(**x),
            }))

    results: List[Result]
    is_food: Optional[bool] = None
    imagecache_id: Optional[str] = None
    _timing: Optional[Timing] = None
    lang: Optional[str] = None
