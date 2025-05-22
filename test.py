from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image
from math import sqrt
import json

recognizer = RecognitionPredictor()
detection = DetectionPredictor()

key_benchmark = [[0.07969151670951156, 0.18337264150943397], [0.18723221936589546, 0.18337264150943397], [0.18723221936589546, 0.19929245283018868], [0.07969151670951156, 0.19929245283018868]]

def normalizer(bboxes, W, H): 
    return [[x / W, y / H] for x, y in bboxes]

def original(bboxes, W, H): 
    return [[x * W, y * H] for x, y in bboxes]

def compute_centroid(polygon):
    x = sum([pt[0] for pt in polygon]) / len(polygon)
    y = sum([pt[1] for pt in polygon]) / len(polygon)
    return (x, y)

def euclidean_distance(c1, c2):
    return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# def find_nearest_polygon(benchmark_polygon, polygon_list, X, Y):
#     benchmark_centroid = compute_centroid(benchmark_polygon)
#     min_distance = float("inf")
#     closest_index = -1
#     for i, polygon in enumerate(polygon_list):
#         polygon = normalizer(polygon, X, Y)
#         poly_centroid = compute_centroid(polygon)
#         dist = euclidean_distance(benchmark_centroid, poly_centroid)
#         if dist < min_distance:
#             min_distance = dist
#             closest_index = i
#     polygon_list.pop(closest_index)
#     return closest_index, polygon_list[closest_index], min_distance, polygon_list

def remove_boxes(reference_polygon, candidate_polygons, below_threshold_px=50, above_threshold_px=5):
    reference_bottom = max(y for x, y in reference_polygon)
    remaining = []

    for poly in candidate_polygons:
        min_y = min(y for x, y in poly)
        max_y = max(y for x, y in poly)
        if (min_y < reference_bottom + below_threshold_px) and (max_y > reference_bottom - above_threshold_px):
            remaining.append(poly)

    return remaining

def find_right_nearest_boxes(reference_polygon, candidate_polygons, k=2):
    # Get right edge x (max x of reference polygon)
    ref_right = max(x for x, y in reference_polygon)
    ref_top = min(y for x, y in reference_polygon)
    ref_bottom = max(y for x, y in reference_polygon)

    def horizontal_distance(polygon):
        # Distance from left edge of candidate to right edge of reference
        poly_left = min(x for x, y in polygon)
        poly_top = min(y for x, y in polygon)
        poly_bottom = max(y for x, y in polygon)
        
        # Penalize if not vertically aligned (no overlap)
        vertical_overlap = not (poly_bottom < ref_top or poly_top > ref_bottom)
        penalty = 1e5 if not vertical_overlap else 0
        return (poly_left - ref_right) + penalty

    # Filter out boxes that are to the left of the reference
    right_candidates = [poly for poly in candidate_polygons if min(x for x, y in poly) > ref_right]

    # Sort by horizontal distance
    sorted_candidates = sorted(right_candidates, key=horizontal_distance)

    return sorted_candidates[:k]

def fetch_result(results, key_polygon, polygons):
    name_polygon = polygons[0]
    address_polygon = polygons[1]
    if min(x for x, y in polygons[0]) < max(x for x, y in polygons[0]):
        name_polygon = polygons[1]
        address_polygon = polygons[0]
    key_value = [value for value in results if value['polygon']==key_polygon]
    name_value = [value for value in results if value['polygon']==name_polygon]
    address_value = [value for value in results if value['polygon']==address_polygon]

    return key_value[0]['text']+": "+name_value[0]['text']+'\n'+address_value[0]['text']

def get_values(image):
    X, Y = image.size
    predictions = recognizer([image], ['en,bn'.split(",")], det_predictor=detection)
    results = []
    for line in predictions[0].text_lines:
        results.append(
            {
                "polygon": line.polygon, 
                "confidence": line.confidence, 
                "text": line.text
            }
        )
    # with open("test_data/desco_output.json", 'r', encoding = 'utf-8') as f: 
    #     results = json.load(f)
    print(f">>>> RESULTS: {results}")
    polygons = [line['polygon'] for line in results]
    # key_value_index, _, _, polygons = find_nearest_polygon(key_benchmark, polygons, X, Y)
    key_value_index = None
    for idx, result in enumerate(results):
        if "Name & Address" in result['text']:
            key_value_index = idx
    if key_value_index is None:
        return
    polygons.pop(key_value_index)
    ref_polygon = results[key_value_index]['polygon']
    print(f">>>> REF: {ref_polygon}")
    new_polygons = remove_boxes(ref_polygon, polygons)
    result_boxes = find_right_nearest_boxes(ref_polygon, new_polygons)

    return fetch_result(results, ref_polygon, result_boxes)

if __name__ == "__main__": 
    image = Image.open("test_data/15998253478_77583667ac_z.jpg").convert("RGB")
    image_2 = Image.open("test_data/photo_2025-05-22_13-04-22.jpg").convert("RGB")
    image_3 = Image.open("test_data/photo_2025-05-22_13-04-30.jpg").convert("RGB")

    result = get_values(image)
    print(result)
    
    result = get_values(image_2)
    print(result)
    
    result = get_values(image_3)
    print(result)
    