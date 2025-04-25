import requests
from bs4 import BeautifulSoup
import numpy as np
import cv2
from io import BytesIO
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import imagehash
import random
from tqdm import tqdm
from pathlib import Path
import math
import sys
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

class DeepScrape:
    def __init__(self, output_dir="Dataset", image_size=(512, 512), search_engine="Google", use_icrawler=True):
        self.output_dir = output_dir
        self.image_size = image_size
        self.search_engine = search_engine
        self.use_icrawler = use_icrawler
        self.num_images = 0
        self.downloaded_hashes = set()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def build_dataset(self, queries, images_per_query=20, rate_limit=1.5):

        if self.use_icrawler:
            if self.search_engine == "Google":
                for i, query in enumerate(queries):
                    print(f"\nProcessing query {i + 1}/{len(queries)}: {query}")
                    save_dir = os.path.join(self.output_dir, query.replace(" ", "_")[:30])
                    os.makedirs(save_dir, exist_ok=True)
                    google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
                    google_crawler.crawl(keyword=query, max_num=images_per_query)
                    print(f"Downloaded {images_per_query} images for '{query}'")

            elif self.search_engine == "Bing":
                for i, query in enumerate(queries):
                    print(f"\nProcessing query {i + 1}/{len(queries)}: {query}")
                    save_dir = os.path.join(self.output_dir, query.replace(" ", "_")[:30])
                    os.makedirs(save_dir, exist_ok=True)
                    bing_crawler = BingImageCrawler(storage={'root_dir': save_dir})
                    bing_crawler.crawl(keyword=query, max_num=images_per_query)
                    print(f"Downloaded {images_per_query} images for '{query}'")
            else:
                raise ValueError("Invalid search engine. Use 'Google' or 'Bing'")

            print(f"{self.output_dir} -saved!")
            return self.output_dir

        for i, query in enumerate(queries):
            print(f"\nProcessing query {i + 1}/{len(queries)}: {query}")

            # Download images for this query
            images, metadata = self._download_images(
                query,
                limit=images_per_query,
                rate_limit=rate_limit
            )

            print(f"Downloaded {len(images)} images for '{query}'")

            if images:
                images = self._filter_duplicates(images)
                self.num_images += len(images)

                # Save images
                save_dir = os.path.join(self.output_dir, query.replace(" ", "_")[:30])
                os.makedirs(save_dir, exist_ok=True)

                for idx, image in enumerate(images):
                    cv2.imwrite(os.path.join(save_dir, f"{idx:0{int(math.log10(images_per_query))}d}.png"),
                                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            time.sleep(3)

        print(f"\nimages: {self.num_images}")
        print(f"{self.output_dir} -saved!")

        return self.output_dir

    def _download_images(self, query, limit=20, rate_limit=1.0):
        base_url = "https://www.bing.com/images/async"

        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.bing.com/",
            "DNT": "1",
        }

        params = {
            "q": query,
            "qft": "+filterui:imagesize-large+filterui:photo-photo",  # Filter for large photos only
            "form": "IRFLTR",
            "first": 0,
            "count": 35
        }

        images_list = []
        metadata_list = []
        error_count = 0

        # Create a progress bar
        pbar = tqdm(total=limit, desc=f"Query: {query[:30]}...", file=sys.stdout)

        while len(images_list) < limit and error_count < 5:
            try:
                # Add rate limiting
                time.sleep(rate_limit)

                # Rotate user agent
                headers["User-Agent"] = random.choice(self.user_agents)

                response = requests.get(base_url, headers=headers, params=params, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                image_containers = soup.find_all("a", class_="iusc")

                if not image_containers:
                    error_count += 1
                    params["first"] += params["count"]
                    continue

                image_urls_info = []

                for container in image_containers:
                    if len(images_list) >= limit:
                        break

                    try:
                        m = container.get("m")
                        if m:
                            image_data = json.loads(m.replace("'", "\""))
                            image_url = image_data.get("murl", "")

                            if not image_url:
                                continue

                            url_info = {
                                "url": image_url,
                                "query": query,
                                "title": image_data.get("t", ""),
                                "source": image_data.get("purl", ""),
                                "date_scraped": time.strftime("%Y-%m-%d %H:%M:%S")
                            }

                            image_urls_info.append(url_info)

                    except Exception as e:
                        continue

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(self._fetch_image, info["url"], headers, info)
                               for info in image_urls_info]

                    for future in as_completed(futures):
                        img_processed, url_info, error = future.result()

                        if img_processed is not None and url_info is not None:
                            if url_info["phash"] in self.downloaded_hashes:
                                continue

                            self.downloaded_hashes.add(url_info["phash"])
                            images_list.append(img_processed)
                            metadata_list.append(url_info)
                            pbar.update(1)

                            if len(images_list) >= limit:
                                break

                if len(images_list) < limit:
                    params["first"] += params["count"]

            except Exception as e:
                error_count += 1
                time.sleep(2)

        pbar.close()
        return images_list, metadata_list

    def _fetch_image(self, image_url, headers, url_info, retry_count=2):

        for attempt in range(retry_count + 1):
            try:
                img_response = requests.get(image_url, headers=headers, timeout=10)
                img_response.raise_for_status()

                try:
                    pil_img = Image.open(BytesIO(img_response.content))
                    # Calculate perceptual hash for deduplication
                    phash = str(imagehash.phash(pil_img))

                    img_array = np.array(pil_img)

                    if len(img_array.shape) == 2:  # Grayscale
                        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    elif img_array.shape[2] == 4:  # With alpha channel
                        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    else:
                        img_rgb = img_array

                    # Skip small images
                    height, width = img_rgb.shape[:2]
                    if width < 512 or height < 512:
                        return None, None, "Image too small"

                    # Assess quality
                    quality_pass, quality_note = self._assess_quality(img_rgb)
                    if not quality_pass:
                        return None, None, quality_note

                    # Preprocess image
                    img_processed = self._preprocess_image(img_rgb)

                    # Update metadata
                    url_info.update({
                        "original_width": width,
                        "original_height": height,
                        "phash": phash,
                        "quality_note": quality_note
                    })

                    return img_processed, url_info, None

                except Exception as e:
                    if attempt == retry_count:
                        return None, None, f"PIL processing error: {str(e)}"
                    time.sleep(1)

            except requests.exceptions.RequestException as e:
                if attempt == retry_count:
                    return None, None, f"Request error: {str(e)}"
                time.sleep(1)

        return None, None, "Maximum retries exceeded"

    def _assess_quality(self, img):

        # Check if image is too dark or too bright
        avg_brightness = np.mean(img)
        if avg_brightness < 30 or avg_brightness > 225:
            return False, "Brightness issue"

        # Check if image has enough contrast
        contrast = np.std(img)
        if contrast < 20:
            return False, "Low contrast"

        # Check if image is mostly a single color (like placeholder images)
        if contrast < 30 and (np.max(img) - np.min(img) < 50):
            return False, "Likely placeholder image"

        # Simple blur detection
        laplacian = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if laplacian < 100:
            return False, "Blurry image"

        return True, "Acceptable quality"

    def _preprocess_image(self, img, preserve_aspect_ratio=True):

        if img is None:
            return None

        target_size = self.image_size

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # With alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Preserve aspect ratio if requested
        if preserve_aspect_ratio:
            h, w = img.shape[:2]
            if h > w:
                new_h, new_w = target_size[0], int(w * target_size[0] / h)
            else:
                new_h, new_w = int(h * target_size[1] / w), target_size[1]

            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a black canvas of target size
            canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            # Compute position to paste resized image
            x_offset = (target_size[1] - new_w) // 2
            y_offset = (target_size[0] - new_h) // 2

            # Paste the resized image
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
            return canvas
        else:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    def _filter_duplicates(self, images, threshold=5):

        if not images:
            return [], []

        # Convert list of NumPy arrays to list of PIL Images
        pil_images = [Image.fromarray(img) for img in images]

        # Calculate perceptual hashes
        hashes = [imagehash.phash(img) for img in pil_images]

        # Identify duplicates
        duplicates = set()
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if hashes[i] - hashes[j] < threshold:
                    duplicates.add(j)

        # Keep only non-duplicate images and metadata
        valid_indices = [i for i in range(len(images)) if i not in duplicates]

        filtered_images = [images[i] for i in valid_indices]

        if len(duplicates) > 0:
            print(f"Removed {len(duplicates)} duplicate images")

        return filtered_images

queries = [
    "modern house exterior", "contemporary house facade", "traditional house exterior", "rustic house facade",
    "minimalist house exterior", "luxury mansion exterior", "glass house exterior",
    "wooden house exterior", "brick house facade", "metal-clad house exterior", "concrete house facade",
    "white house exterior", "black house facade", "colorful house exterior", "industrial house exterior",
    "eco-friendly house exterior", "green roof house", "sustainable house design", "solar-powered house exterior"
]

queries += [
    "Victorian house exterior", "Georgian house facade", "Colonial house exterior", "Tudor house facade",
    "Mediterranean house exterior", "Spanish villa facade", "Italian villa exterior", "Japanese house exterior",
    "Scandinavian house facade", "Nordic house exterior", "Art Deco building facade",
    "Mid-century modern house exterior",
    "Futuristic house exterior", "Gothic architecture exterior", "Baroque building facade",
    "Neoclassical building exterior"
]

queries += [
    "brick house exterior", "stone house facade", "wood cladding house", "metal facade house",
    "glass and steel house exterior", "stucco house facade", "concrete house exterior",
    "Corten steel house exterior", "rammed earth house",
    "exposed concrete architecture", "fiber cement facade", "shingle-clad house",
    "clay tile house roof", "slate roof house", "cob house exterior", "thatched roof cottage"
]

queries += [
    "house in the mountains", "beachfront house exterior", "desert house architecture",
    "tropical house exterior", "urban house facade", "suburban house exterior",
    "countryside house architecture", "forest cabin exterior", "lake house exterior",
    "cliffside house design", "floating house architecture", "off-grid house exterior",
    "tiny house exterior", "prefabricated house facade", "shipping container house exterior"
]

queries += [
    "modern skyscraper exterior", "glass tower facade", "office building exterior",
    "high-rise residential building facade", "futuristic skyscraper", "brutalist architecture facade",
    "sustainable office building exterior", "biophilic architecture facade", "mixed-use building exterior",
    "retail building facade", "hotel building exterior", "airport terminal architecture",
    "museum building facade", "theater exterior design", "library building facade",
    "university campus architecture", "stadium exterior design"
]

# Architectural Styles
queries += [
    "Craftsman house exterior", "Prairie style house facade", "Ranch house exterior", "Split-level house facade",
    "Bauhaus architecture exterior", "Postmodern house facade", "Art Nouveau building exterior", "Brutalist house design",
    "Cape Cod house exterior", "Queen Anne Victorian facade", "Edwardian house exterior", "Federal style house facade",
    "French Provincial house exterior", "Greek Revival facade", "Gothic Revival house exterior", "Italianate house facade",
    "Mission Revival exterior", "Pueblo Revival house", "Romanesque architecture exterior", "Second Empire house facade",
    "Shingle style house exterior", "Southwest adobe house", "Swiss chalet exterior", "A-frame house facade",
    "Barn house conversion exterior", "Bungalow house facade", "Cabin exterior design", "Charleston single house facade",
    "Dutch Colonial exterior", "English Cottage facade", "Farmhouse exterior design", "Garrison Colonial house facade"
]

# Material Combinations
queries += [
    "brick and stone house exterior", "wood and glass house facade", "stucco and wood exterior",
    "brick and siding house facade", "stone and timber exterior", "glass and concrete house facade",
    "stone and stucco exterior", "timber and steel house facade", "bamboo and concrete exterior",
    "limestone and glass facade", "marble and wood exterior", "granite and glass house facade",
    "slate and timber exterior", "sandstone and wood facade", "cedar and concrete exterior",
    "limestone and brick facade", "quartz and timber exterior", "travertine facade house",
    "copper-clad house exterior", "zinc facade house", "aluminum cladding exterior",
    "weathered steel house facade", "polycarbonate and wood exterior", "recycled material house facade"
]

# Color-Focused
queries += [
    "blue house exterior", "red brick house facade", "yellow house exterior", "green house facade",
    "gray stone house exterior", "terracotta house facade", "navy blue house exterior", "cream colored house facade",
    "pastel colored house exterior", "earth tone house facade", "monochromatic house exterior", "two-tone house facade",
    "coral colored house exterior", "mint green house facade", "lavender house exterior", "turquoise house facade",
    "salmon colored house exterior", "olive green house facade", "burgundy house exterior", "sage green house facade",
    "teal house exterior", "charcoal house facade", "taupe colored house exterior", "beige house facade"
]

# Regional/Country-Specific
queries += [
    "Australian beach house exterior", "American farmhouse facade", "Canadian log house exterior", "Chinese courtyard house facade",
    "Dutch townhouse exterior", "English country cottage facade", "French chateau exterior", "German fachwerk house facade",
    "Greek island house exterior", "Hawaiian plantation house facade", "Indian bungalow exterior", "Irish stone cottage facade",
    "Italian palazzo exterior", "Moroccan riad facade", "New Zealand beach house exterior", "Norwegian wooden house facade",
    "Portuguese villa exterior", "Russian dacha facade", "Scottish stone house exterior", "South African Cape Dutch facade",
    "Swedish summer house exterior", "Swiss mountain chalet facade", "Thai stilt house exterior", "Turkish stone house facade"
]

# Feature-Focused
queries += [
    "house with wrap-around porch exterior", "house with balcony facade", "house with courtyard exterior",
    "house with rooftop garden facade", "house with large windows exterior", "house with double-height windows facade",
    "house with dormer windows exterior", "house with bay windows facade", "house with turret exterior",
    "house with columns facade", "house with arched doorway exterior", "house with gable roof facade",
    "house with hipped roof exterior", "house with flat roof facade", "house with curved walls exterior",
    "house with cantilevered sections facade", "house with sliding glass walls exterior", "house with louvered facade",
    "house with underground garage exterior", "house with breezeway facade", "house with outdoor fireplace exterior",
    "house with infinity pool facade", "house with greenhouse exterior", "house with observatory dome facade"
]

# Landscape Integration
queries += [
    "house integrated with hillside", "terraced house on slope exterior", "house built around trees facade",
    "house with green wall exterior", "house with landscaped roof facade", "house with waterfall integration exterior",
    "house built over water facade", "house with rock outcropping exterior", "house with internal courtyard garden facade",
    "house with desert landscaping exterior", "house with tropical garden facade", "house with vertical garden exterior",
    "house with zen garden facade", "house with English garden exterior", "house with native landscaping facade",
    "house with wildflower meadow exterior", "house with lavender field facade", "house with olive grove exterior",
    "house with vineyard facade", "house with orchard exterior", "house with bamboo grove facade",
    "house with coastal dune landscaping", "house with alpine garden exterior", "house with rainforest integration facade"
]

# Commercial and Public Buildings
queries += [
    "modern restaurant exterior", "boutique hotel facade", "school building exterior", "hospital facade design",
    "fire station exterior", "police station facade", "government building exterior", "courthouse facade",
    "post office building exterior", "train station facade", "bus terminal exterior", "shopping mall facade",
    "retail park exterior", "office park facade", "industrial complex exterior", "warehouse conversion facade",
    "community center exterior", "recreational center facade", "sports complex exterior", "theater building facade",
    "concert hall exterior", "art gallery facade", "museum building exterior", "library architecture facade",
    "church building exterior", "temple facade design", "mosque exterior", "synagogue facade"
]

# Contemporary Trends
queries += [
    "passive house exterior", "net-zero energy house facade", "carbon-neutral house exterior",
    "3D printed house facade", "parametric house design exterior", "modular house facade",
    "stackable housing units exterior", "micro apartment building facade", "co-living space exterior",
    "smart house facade", "IoT integrated house exterior", "self-sufficient house facade",
    "farm-to-table house exterior", "vertical farming integrated building facade", "rainwater harvesting house exterior",
    "geothermal house facade", "wind-powered house exterior", "biomimicry architecture facade",
    "living building exterior", "circular economy house facade", "upcycled material house exterior",
    "biodegradable house facade", "adaptable house exterior", "demountable house facade"
]

# Mixed Use and Urban
queries += [
    "live-work building exterior", "shophouse facade", "mixed-use tower exterior", "urban infill house facade",
    "townhouse row exterior", "brownstone facade", "loft building exterior", "converted warehouse facade",
    "urban courtyard house exterior", "city terrace house facade", "apartment complex exterior",
    "condominium building facade", "urban villa exterior", "penthouse terrace facade", "rooftop extension exterior",
    "urban micro house facade", "vertical village exterior", "sky garden tower facade", "pocket neighborhood exterior",
    "cooperative housing facade", "community land trust housing exterior", "urban cohousing facade",
    "accessible housing exterior", "multigenerational house facade", "flexhouse exterior", "courtyard apartment building facade"
]

# Historical and Period-Specific
queries += [
    "1920s house exterior", "1930s Art Deco house facade", "1940s suburban house exterior", "1950s ranch house facade",
    "1960s modernist house exterior", "1970s house facade", "1980s contemporary house exterior", "1990s house facade",
    "Antebellum house exterior", "Civil War era house facade", "Gilded Age mansion exterior", "Industrial Revolution factory conversion facade",
    "Medieval style house exterior", "Renaissance inspired villa facade", "Byzantine style house exterior", "Ottoman inspired house facade",
    "Ancient Greek style house exterior", "Roman villa inspired facade", "Ancient Egyptian style house exterior", "Mayan inspired house facade",
    "Chinese dynasty inspired house exterior", "Edo period Japanese house facade", "Belle Époque mansion exterior", "Regency style house facade"
]

# Specialized Types
queries += [
    "treehouse exterior design", "underground earth house facade", "dome house exterior", "yurt-inspired house facade",
    "lighthouse conversion exterior", "windmill conversion facade", "barn conversion exterior", "church conversion facade",
    "industrial loft conversion exterior", "school conversion facade", "water tower conversion exterior", "grain silo home facade",
    "railway carriage house exterior", "boathouse facade", "houseboat exterior", "floating house facade",
    "ice hotel exterior", "desert earthship facade", "tropical pavilion house exterior", "arctic research station facade",
    "mountain observatory exterior", "beach hut facade", "safari lodge exterior", "wilderness retreat facade",
    "disaster-resistant house exterior", "hurricane-proof house facade", "earthquake-resistant building exterior", "flood-resistant house facade"
]

# Light and Transparency
queries += [
    "transparent house exterior", "translucent house facade", "house with light well exterior", "house with internal courtyard facade",
    "house with skylight exterior", "house with clerestory windows facade", "house with light shelf exterior", "house with glass blocks facade",
    "house with light tunnel exterior", "house with reflective facade", "house with filtered light exterior", "house with shadow play facade",
    "house with dappled light exterior", "house with brise soleil facade", "house with perforated screen exterior", "house with light canon facade",
    "house with oriented windows exterior", "house with strategic lighting facade", "house with ambient light exterior", "house with dramatic lighting facade",
    "house with night illumination exterior", "house with LED integrated facade", "house with fiber optic lighting exterior", "house with dynamic lighting facade"
]

# Specific Roof Styles
queries += [
    "mansard roof house exterior", "gambrel roof house facade", "butterfly roof house exterior", "shed roof house facade",
    "sawtooth roof house exterior", "barrel roof house facade", "conical roof house exterior", "pyramidal roof house facade",
    "dome roof house exterior", "clerestory roof house facade", "monitor roof house exterior", "saltbox roof house facade",
    "jerkinhead roof house exterior", "half-hipped roof house facade", "dutch gable roof house exterior", "bonnet roof house facade",
    "skillion roof house exterior", "M-shaped roof house facade", "cross gable roof house exterior", "cross hipped roof house facade",
    "combination roof house exterior", "curved roof house facade", "parapet roof house exterior", "crown roof house facade",
    "folded plate roof house exterior", "geodesic dome roof house facade", "lean-to roof house exterior", "pagoda style roof house facade"
]

# Window and Door Features
queries += [
    "house with circular windows exterior", "house with ribbon windows facade", "house with picture windows exterior", "house with casement windows facade",
    "house with transom windows exterior", "house with jalousie windows facade", "house with stained glass windows exterior", "house with lattice windows facade",
    "house with double-hung windows exterior", "house with oriel windows facade", "house with pivot windows exterior", "house with french doors facade",
    "house with dutch doors exterior", "house with pivot doors facade", "house with sliding doors exterior", "house with bi-fold doors facade",
    "house with barn doors exterior", "house with pocket doors facade", "house with glass garage doors exterior", "house with carriage doors facade",
    "house with porthole windows exterior", "house with hopper windows facade", "house with awning windows exterior", "house with plantation shutters facade",
    "house with glass block windows exterior", "house with floor-to-ceiling windows facade", "house with mullioned windows exterior", "house with fanlight windows facade"
]

# Property Types
queries += [
    "single-family detached house exterior", "duplex house facade", "triplex house exterior", "quadplex house facade",
    "row house exterior", "terraced house facade", "semi-detached house exterior", "garden apartment facade",
    "maisonette exterior", "patio home facade", "cluster home exterior", "zero-lot-line house facade",
    "coach house exterior", "carriage house facade", "guest house exterior", "pool house facade",
    "gatehouse exterior", "caretaker's cottage facade", "granny flat exterior", "in-law suite facade",
    "laneway house exterior", "backyard cottage facade", "casita exterior", "accessory dwelling unit facade",
    "mews house exterior", "courtyard house facade", "atrium house exterior", "patio house facade"
]

# Climate-Specific
queries += [
    "tropical climate house exterior", "subtropical house facade", "mediterranean climate house exterior", "desert climate house facade",
    "arid region house exterior", "semi-arid house facade", "temperate climate house exterior", "continental climate house facade",
    "humid continental house exterior", "oceanic climate house facade", "subarctic house exterior", "arctic house facade",
    "alpine house exterior", "monsoon region house facade", "rainforest house exterior", "savanna climate house facade",
    "steppe house exterior", "tundra house facade", "permafrost region house exterior", "coastal hurricane-resistant house facade",
    "tornado alley house exterior", "earthquake zone house facade", "avalanche-resistant mountain house exterior", "typhoon-resistant house facade",
    "flood plain elevated house exterior", "wetland stilt house facade", "high altitude house exterior", "high humidity house facade"
]

# Compound Structures
queries += [
    "house with detached garage exterior", "house with attached carport facade", "house with porte-cochere exterior", "house with breezeway connection facade",
    "house with courtyard garage exterior", "house with underground garage facade", "house with motor court exterior", "house with circular driveway facade",
    "house with separate guest pavilion exterior", "house with connected studio facade", "house with workshop annex exterior", "house with garden shed facade",
    "house with greenhouse attachment exterior", "house with conservatory facade", "house with solarium exterior", "house with orangery facade",
    "house with pool cabana exterior", "house with outdoor kitchen pavilion facade", "house with gazebo exterior", "house with pergola facade",
    "house with belvedere exterior", "house with observation tower facade", "house with rooftop deck exterior", "house with covered terrace facade",
    "house with exterior staircase exterior", "house with ramped entry facade", "house with bridge connection exterior", "house with multiple wings facade"
]

# Unique Layouts
queries += [
    "L-shaped house exterior", "U-shaped house facade", "H-shaped house exterior", "T-shaped house facade",
    "cross-shaped house exterior", "octagonal house facade", "hexagonal house exterior", "circular house facade",
    "oval house exterior", "triangular house facade", "zigzag house exterior", "stepped house facade",
    "cascading house exterior", "terraced house on hillside facade", "split level house exterior", "sunken living room house facade",
    "raised main floor house exterior", "open plan house facade", "clustered volumes house exterior", "pavilion style house facade",
    "courtyard house exterior", "donut-shaped house facade", "radial plan house exterior", "pinwheel layout house facade",
    "spiral layout house exterior", "interlocking volumes house facade", "indoor-outdoor house exterior", "layered house facade"
]

# Specific Era Modern
queries += [
    "1990s postmodern house exterior", "early 2000s McMansion facade", "2010s minimalist house exterior", "2020s sustainability-focused house facade",
    "late modern architecture house exterior", "neo-futurist house facade", "neomodernist house exterior", "parametric design house facade",
    "deconstructivist house exterior", "high-tech architecture house facade", "critical regionalism house exterior", "metabolist architecture house facade",
    "structuralist house exterior", "expressionist architecture house facade", "form follows function house exterior", "neo-expressionist house facade",
    "neo-vernacular house exterior", "organic modernism house facade", "digital architecture house exterior", "blob architecture house facade",
    "folding architecture house exterior", "liquid architecture house facade", "topological architecture house exterior", "morphogenetic design house facade",
    "neo-constructivist house exterior", "diagrammatic architecture house facade", "pragmatist house exterior", "sustainable modernism house facade"
]

# Cultural and Religious
queries += [
    "Balinese style house exterior", "Moroccan courtyard house facade", "Thai pavilion style house exterior", "African compound house facade",
    "Russian dacha exterior", "Irish cottage facade", "New England saltbox exterior", "Southwestern adobe house facade",
    "Mexican hacienda exterior", "Bavarian chalet facade", "Alpine mountain house exterior", "Indonesian longhouse facade",
    "Filipino bahay kubo exterior", "Maori meeting house facade", "Buddhist temple-inspired house exterior", "Hindu temple-inspired house facade",
    "Islamic architecture-inspired house exterior", "Shinto shrine-inspired house facade", "Gothic cathedral-inspired house exterior", "Baroque church-inspired house facade",
    "Byzantine-inspired house exterior", "Orthodox church-inspired house facade", "Pueblo mission-inspired house exterior", "Quaker meetinghouse-inspired facade",
    "Monastery-inspired house exterior", "Abbey-inspired house facade", "Synagogue-inspired house exterior", "Mosque-inspired house facade"
]

# Emerging Technologies
queries += [
    "3D-printed concrete house exterior", "robotic fabricated house facade", "mass timber house exterior", "cross-laminated timber house facade",
    "digital fabrication house exterior", "computationally designed house facade", "algorithmically optimized house exterior", "AI-designed house facade",
    "generative design house exterior", "topology optimized house facade", "carbon-sequestering material house exterior", "mycelium-based material house facade",
    "self-healing concrete house exterior", "phase-changing material house facade", "photocatalytic surface house exterior", "electrochromic glass house facade",
    "thermochromic material house exterior", "kinetic facade house", "responsive architecture house exterior", "sensor-embedded house facade",
    "data-driven design house exterior", "digitally-fabricated panel house facade", "component-based assembly house exterior", "precision-manufactured house facade",
    "drone-constructed house exterior", "robot-built house facade", "augmented reality integrated house exterior", "virtual reality designed house facade"
]

print(f"Total number of queries: {len(queries)}")
print("Building dataset of size:", len(queries) * 1)
builder = DeepScrape(output_dir="Dataset",
                     search_engine="Google",
                     use_icrawler=True)

dataset_path = builder.build_dataset(queries=queries, images_per_query=1)