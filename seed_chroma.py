"""Build a small ChromaDB collection of sample products for the FrontierAgent's RAG.

The production pipeline uses ~400k Amazon product embeddings; this script seeds
a compact 60-item collection that's enough to demo the agent locally.

Run once before launching the demo:
    python seed_chroma.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "products_vectorstore"
COLLECTION = "products"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_PRODUCTS = [
    ("Apple AirPods Pro 2nd generation with active noise cancellation, MagSafe charging case, USB-C", "Electronics", 199.0),
    ("Apple iPad 10th generation Wi-Fi 64GB tablet with 10.9 inch Liquid Retina display A14 Bionic chip", "Electronics", 349.0),
    ("Apple iPhone 14 Pro Max 256GB unlocked smartphone with A16 Bionic and ProMotion display", "Electronics", 999.0),
    ("Apple MacBook Air M2 13.6 inch laptop 8GB RAM 256GB SSD Liquid Retina display", "Electronics", 1099.0),
    ("Apple Watch Series 9 GPS 45mm aluminum case with sport band always-on display", "Electronics", 429.0),
    ("Samsung Galaxy S24 Ultra 256GB unlocked smartphone with Snapdragon 8 Gen 3 and S Pen", "Electronics", 1199.0),
    ("Samsung Galaxy Watch Ultra 47mm LTE titanium smartwatch with health tracking and GPS", "Electronics", 649.0),
    ("Sony WH-1000XM5 wireless noise cancelling over-ear headphones with 30 hour battery", "Electronics", 399.0),
    ("Bose QuietComfort Ultra wireless headphones with spatial audio and active noise cancellation", "Electronics", 429.0),
    ("Sonos Era 300 wireless smart speaker with Dolby Atmos spatial audio and AirPlay 2", "Electronics", 449.0),
    ("Hisense 55 inch R6 Series 4K UHD Roku smart TV with Dolby Vision HDR and three HDMI ports", "Electronics", 248.0),
    ("LG C3 65 inch OLED evo 4K smart TV with Alpha 9 Gen6 AI processor and webOS 23", "Electronics", 1799.0),
    ("Samsung Frame QLED 55 inch 4K smart TV that doubles as art when off with anti-glare matte display", "Electronics", 999.0),
    ("Roku Ultra 4K HDR streaming player with Dolby Vision Atmos and voice remote pro", "Electronics", 99.0),
    ("Amazon Echo Show 10 3rd gen smart display with motion 10.1 inch HD screen and Alexa", "Electronics", 249.0),
    ("Google Nest Hub Max 10 inch smart display with built-in camera and Google Assistant", "Electronics", 229.0),
    ("Ring Video Doorbell Pro 2 with 1536p HD video and 3D motion detection", "Smart Home", 249.0),
    ("Nest Learning Thermostat 3rd generation that programs itself to save energy", "Smart Home", 249.0),
    ("Philips Hue White and Color Ambiance starter kit with bridge and four E26 bulbs", "Smart Home", 199.0),
    ("Eufy RoboVac X8 Pro robot vacuum with twin turbine technology and AI obstacle avoidance", "Smart Home", 549.0),
    ("iRobot Roomba j7 Plus self-emptying robot vacuum with PrecisionVision navigation", "Smart Home", 799.0),
    ("Dyson V15 Detect cordless stick vacuum with laser slim fluffy cleaner head", "Appliances", 749.0),
    ("Shark NV356E Navigator Lift-Away upright vacuum with HEPA filter and swivel steering", "Appliances", 199.0),
    ("Instant Pot Duo 7-in-1 6 quart electric pressure cooker slow cooker rice cooker steamer", "Appliances", 99.0),
    ("Ninja Foodi 6.5 quart pressure cooker with TenderCrisp technology and air fryer lid", "Appliances", 199.0),
    ("Breville Barista Express espresso machine with built-in conical burr grinder", "Appliances", 749.0),
    ("KitchenAid Artisan tilt-head 5 quart stand mixer with 10 speeds and pouring shield", "Appliances", 449.0),
    ("Vitamix 5200 professional grade blender with 64oz container and self-cleaning", "Appliances", 549.0),
    ("Dell XPS 15 9530 laptop Intel Core i7-13700H 16GB RAM 512GB SSD 15.6 inch OLED", "Computers", 1899.0),
    ("Dell G15 gaming laptop AMD Ryzen 5 7640HS 16GB RAM 1TB SSD GeForce RTX 3050 6GB", "Computers", 899.0),
    ("Lenovo IdeaPad Slim 5 16 inch touch laptop AMD Ryzen 5 8645HS 16GB RAM 512GB SSD", "Computers", 649.0),
    ("HP Pavilion 15 laptop AMD Ryzen 7 7730U 16GB RAM 512GB SSD with backlit keyboard", "Computers", 749.0),
    ("ASUS ROG Strix G16 gaming laptop Intel Core i7-13650HX 16GB RAM 1TB SSD RTX 4060", "Computers", 1499.0),
    ("Microsoft Surface Pro 9 13 inch tablet Intel Core i5 8GB RAM 256GB SSD with type cover", "Computers", 1099.0),
    ("Logitech MX Master 3S wireless mouse with quiet click and 8000 DPI tracking", "Computers", 99.0),
    ("Logitech MX Keys S advanced wireless illuminated keyboard with smart actions", "Computers", 109.0),
    ("Dell UltraSharp U2723QE 27 inch 4K USB-C hub monitor with 90W power delivery", "Computers", 649.0),
    ("LG UltraGear 27GP950-B 27 inch Nano IPS 4K 144Hz HDR600 gaming monitor", "Computers", 799.0),
    ("Samsung Odyssey G7 32 inch curved 1440p 240Hz QLED gaming monitor with HDR600", "Computers", 699.0),
    ("Synology DiskStation DS923+ 4-bay NAS with AMD Ryzen R1600 and 4GB DDR4 ECC", "Computers", 599.0),
    ("Western Digital WD Black SN850X 2TB NVMe internal SSD PCIe Gen4 with heatsink", "Computers", 199.0),
    ("Anker 737 GaNPrime 120W three port USB-C wall charger compact travel adapter", "Electronics", 99.0),
    ("Anker 737 PowerCore 24K 24000mAh portable battery with 140W USB-C output", "Electronics", 149.0),
    ("Bose Soundbar 900 with Dolby Atmos voice4video calls and Wi-Fi streaming", "Electronics", 899.0),
    ("Sonos Beam Gen 2 compact smart soundbar with Dolby Atmos and HDMI eARC", "Electronics", 499.0),
    ("DJI Mini 4 Pro folding drone with 4K HDR camera 34 minute flight time and tri-directional obstacle sensing", "Electronics", 999.0),
    ("GoPro Hero 12 Black action camera 5.3K60 video with HyperSmooth 6.0 stabilization", "Electronics", 399.0),
    ("Garmin fenix 7 Pro Sapphire Solar 47mm multisport GPS smartwatch with topo maps", "Electronics", 899.0),
    ("Fitbit Charge 6 advanced fitness tracker with built-in GPS heart rate and Google apps", "Electronics", 159.0),
    ("Theragun Prime percussive deep tissue massage gun with smart app integration", "Health", 299.0),
    ("Hydro Flask 32oz wide mouth insulated stainless steel water bottle with flex straw cap", "Sports", 49.0),
    ("YETI Tundra 45 hard cooler with permafrost insulation and bear resistant design", "Sports", 325.0),
    ("Stanley Quencher H2.0 FlowState 40oz tumbler with handle insulated stainless steel", "Sports", 45.0),
    ("Weber Spirit II E-310 three burner liquid propane gas grill with porcelain enameled grates", "Outdoors", 599.0),
    ("Traeger Pro 575 wood pellet smoker grill with WiFIRE technology and 575 sq inch grilling area", "Outdoors", 899.0),
    ("DeWalt 20V MAX XR cordless drill driver kit with two 5Ah batteries fast charger and bag", "Tools", 249.0),
    ("Milwaukee M18 FUEL cordless impact wrench 1/2 inch high torque with 1400 ft-lb max", "Tools", 399.0),
    ("Bosch Professional 18V GBH18V-26F SDS-plus rotary hammer kit with two 4Ah batteries", "Tools", 449.0),
    ("RYOBI 40V HP brushless 21 inch self-propelled walk-behind lawn mower kit with battery", "Tools", 599.0),
    ("Lego Star Wars UCS Millennium Falcon 75192 collector building set 7541 pieces", "Toys", 849.0),
    ("Nintendo Switch OLED Model handheld gaming console with 7 inch OLED screen 64GB", "Toys", 349.0),
]


def main():
    print(f"Building Chroma collection at {DB_PATH}/{COLLECTION}")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    embedder = SentenceTransformer(EMBED_MODEL)

    documents = [p[0] for p in SAMPLE_PRODUCTS]
    metadatas = [{"category": p[1], "price": p[2]} for p in SAMPLE_PRODUCTS]
    ids = [f"item-{i}" for i in range(len(SAMPLE_PRODUCTS))]

    print(f"Embedding {len(documents)} products with {EMBED_MODEL}...")
    embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Seeded {len(documents)} items into the '{COLLECTION}' collection.")


if __name__ == "__main__":
    main()
