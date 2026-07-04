import numpy as np
from sentence_transformers import SentenceTransformer

product_catalog = [
    {"id": 1, "name": "Wireless Bluetooth Earbuds", "category": "Electronics",
     "description": "Compact true wireless earbuds with active noise cancellation, 24-hour battery life with charging case, and touch controls for music and calls."},

    {"id": 2, "name": "Over-Ear Studio Headphones", "category": "Electronics",
     "description": "Professional over-ear headphones with deep bass, padded ear cups, and a foldable design for studio recording and everyday listening."},

    {"id": 3, "name": "Smartwatch Fitness Tracker", "category": "Electronics",
     "description": "Waterproof smartwatch that tracks heart rate, sleep, steps, and workouts, with smartphone notifications and a week-long battery life."},

    {"id": 4, "name": "Portable Power Bank 20000mAh", "category": "Electronics",
     "description": "High-capacity portable charger with fast charging support for phones and tablets, includes dual USB output ports and LED battery indicator."},

    {"id": 5, "name": "Mechanical Gaming Keyboard", "category": "Electronics",
     "description": "RGB backlit mechanical keyboard with tactile switches, programmable macro keys, and durable aluminum frame for gaming and typing."},

    {"id": 6, "name": "Stainless Steel Water Bottle", "category": "Home & Kitchen",
     "description": "Insulated stainless steel bottle that keeps drinks cold for 24 hours or hot for 12 hours, leak-proof lid, ideal for gym and travel."},

    {"id": 7, "name": "Non-Stick Frying Pan Set", "category": "Home & Kitchen",
     "description": "Set of three non-stick frying pans with heat-resistant handles, suitable for induction, gas, and electric stovetops."},

    {"id": 8, "name": "Electric Kettle 1.7L", "category": "Home & Kitchen",
     "description": "Fast-boiling electric kettle with auto shut-off and boil-dry protection, cordless base for easy pouring."},

    {"id": 9, "name": "Memory Foam Pillow", "category": "Home & Kitchen",
     "description": "Ergonomic memory foam pillow that contours to head and neck for better spinal alignment and improved sleep quality."},

    {"id": 10, "name": "Robot Vacuum Cleaner", "category": "Home & Kitchen",
     "description": "Smart robot vacuum with mapping technology, app control, and automatic recharging, effective on carpets and hard floors."},

    {"id": 11, "name": "Running Shoes for Men", "category": "Sports & Outdoors",
     "description": "Lightweight running shoes with breathable mesh upper and cushioned sole designed for long-distance running and daily training."},

    {"id": 12, "name": "Yoga Mat with Carrying Strap", "category": "Sports & Outdoors",
     "description": "Non-slip yoga mat made from eco-friendly TPE material, extra thick for joint support during yoga and floor exercises."},

    {"id": 13, "name": "Adjustable Dumbbell Set", "category": "Sports & Outdoors",
     "description": "Space-saving adjustable dumbbells that replace multiple weights, ideal for strength training and home gym workouts."},

    {"id": 14, "name": "Camping Tent for 4 People", "category": "Sports & Outdoors",
     "description": "Waterproof camping tent with easy setup, mesh ventilation windows, and enough space for four people and gear."},

    {"id": 15, "name": "Insulated Hiking Backpack", "category": "Sports & Outdoors",
     "description": "Durable hiking backpack with multiple compartments, hydration bladder compatibility, and padded straps for long treks."},

    {"id": 16, "name": "Cotton Crew Neck T-Shirt", "category": "Clothing",
     "description": "Soft breathable cotton t-shirt available in multiple colors, casual fit suitable for everyday wear."},

    {"id": 17, "name": "Slim Fit Denim Jeans", "category": "Clothing",
     "description": "Classic slim fit jeans made from stretch denim fabric, comfortable for daily wear with a modern tapered leg."},

    {"id": 18, "name": "Winter Puffer Jacket", "category": "Clothing",
     "description": "Warm insulated puffer jacket with water-resistant shell, ideal for cold weather and outdoor winter activities."},

    {"id": 19, "name": "Leather Ankle Boots", "category": "Clothing",
     "description": "Genuine leather ankle boots with a comfortable rubber sole, suitable for casual and semi-formal outfits."},

    {"id": 20, "name": "Children's Building Blocks Set", "category": "Toys & Games",
     "description": "Colorful building block set that encourages creativity and motor skills development in children ages 3 and up."},

    {"id": 21, "name": "Board Game for Family Night", "category": "Toys & Games",
     "description": "Strategy-based board game for 2 to 6 players, designed for family game nights with easy-to-learn rules."},

    {"id": 22, "name": "Remote Control Racing Car", "category": "Toys & Games",
     "description": "High-speed remote control car with rechargeable battery, rugged tires, and responsive steering for indoor and outdoor use."},

    {"id": 23, "name": "Air Fryer 5.5L", "category": "Home & Kitchen",
     "description": "Digital air fryer with multiple preset cooking modes, uses little to no oil for healthier fried food at home."},

    {"id": 24, "name": "Noise Cancelling Travel Pillow", "category": "Sports & Outdoors",
     "description": "Memory foam neck pillow with built-in speakers and noise cancellation, designed for comfortable sleep during travel."},

    {"id": 25, "name": "Wireless Charging Pad", "category": "Electronics",
     "description": "Slim wireless charging pad compatible with most smartphones, fast charging support with LED indicator light."},
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
for d in product_catalog:
    d["embedded"]=model.encode(d["description"])

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)

def search(query, top_k = 3):
    query_embedding = model.encode(query)
    results = []
    for product in product_catalog:
        result = {
            "id": product["id"],
            "name": product["name"],
            "category": product["category"],
            "description": product["description"],
            "score": float(cosine_similarity(query_embedding, product["embedded"]))
        }
        results.append(result)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

if __name__ == "__main__":
    query = input("Enter your query: ")
    result = search(query)
    print("Top 3 matched results")
    print("-"*70)
    for r in result:
        print(r)
