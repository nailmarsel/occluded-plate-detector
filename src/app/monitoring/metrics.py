from prometheus_client import Counter, Gauge, Histogram

# --- Service-level counters (step 1) already covered by instrumentator ---
# These are specifically for input data quality monitoring

# Counter for total processed images (search + index)
images_processed_total = Counter(
    "autobahncv_images_processed_total",
    "Total number of images processed",
    ["endpoint"]  # /search or /index
)

# Counter for different failure reasons
input_errors_total = Counter(
    "autobahncv_input_errors_total",
    "Input validation errors",
    ["reason"]  # invalid_format, empty_file, too_large, etc.
)

# Per-neuron failure counters
neuron_failures_total = Counter(
    "autobahncv_neuron_failures_total",
    "Failures at each processing stage",
    ["stage"]  # car_detection, plate_detection, ocr, embedding
)

# Plate detection fallback counter
plate_fallback_total = Counter(
    "autobahncv_plate_fallback_total",
    "Number of times heuristic plate detection was used as fallback"
)

# Confidence histograms for each neuron
confidence_car = Histogram(
    "autobahncv_confidence_car",
    "Confidence of car detection",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

confidence_plate = Histogram(
    "autobahncv_confidence_plate",
    "Confidence of license plate detection",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

confidence_ocr = Histogram(
    "autobahncv_confidence_ocr",
    "Confidence of OCR recognition",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Distribution of recognized plate number length
plate_length = Histogram(
    "autobahncv_plate_length",
    "Length of recognized plate number",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)

# Image size distribution (to detect changes in input quality/resolution)
image_size_bytes = Histogram(
    "autobahncv_image_size_bytes",
    "Size of uploaded images in bytes",
    buckets=[1024, 10240, 102400, 524288, 1048576, 2097152, 5242880, 10485760]
)

# Gauge for the number of empty plates in recent time window (for drift)
empty_plate_rate = Gauge(
    "autobahncv_empty_plate_rate",
    "Rate of images with no plate recognized (over last 5 min)"
)

search_similarity_score = Histogram(
    "autobahncv_search_similarity_score",
    "Similarity score of search results (top-1 score)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

feedback_actions_total = Counter(
    "autobahncv_feedback_actions_total",
    "User feedback actions",
    ["action"]  # confirm, reject, correct, disputed
)
