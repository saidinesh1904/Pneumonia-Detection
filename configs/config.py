
# Dataset

DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"

DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

TRAIN_FOLDER = "train"
VAL_FOLDER = "val"
TEST_FOLDER = "test"


# Image Configuration

IMG_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# DataLoader

BATCH_SIZE = 32

NUM_WORKERS = 2

PIN_MEMORY = True


# Training

SEED = 42

NUM_EPOCHS = 20

PATIENCE = 5

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

FREEZE_BACKBONE = True


# Model

MODEL_NAME = "efficientnet_b0"

NUM_CLASSES = 2

DROPOUT = 0.30


# Learning Rate Scheduler

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 3

MIN_LR = 1e-7


# Grad-CAM

GRADCAM_IMAGES = 10


# Inference

BENCHMARK_RUNS = 100


# Directories

MODELS_DIR = "models"

RESULTS_DIR = "results"

PLOTS_DIR = "results/plots"

METRICS_DIR = "results/metrics"

HISTORY_DIR = "results/history"

PREDICTIONS_DIR = "results/predictions"


# Checkpoints

BEST_MODEL_PATH = "models/best_model.pth"

FINAL_MODEL_PATH = "models/final_model.pth"

HISTORY_PATH = "results/history/training_history.csv"

METRICS_PATH = "results/metrics/metrics.json"

CLASSIFICATION_REPORT_PATH = (
    "results/metrics/classification_report.txt"
)

BENCHMARK_PATH = "results/metrics/benchmark.json"


# Plot Paths

SAMPLE_IMAGES_PATH = (
    "results/plots/sample_images.png"
)

DATASET_STATISTICS_PATH = (
    "results/plots/dataset_statistics.png"
)

SAMPLE_BATCH_PATH = (
    "results/plots/sample_batch.png"
)

TRAINING_CURVES_PATH = (
    "results/plots/training_curves.png"
)

CONFUSION_MATRIX_PATH = (
    "results/plots/confusion_matrix.png"
)

ROC_PR_CURVES_PATH = (
    "results/plots/roc_pr_curves.png"
)

PREDICTIONS_PATH = (
    "results/plots/prediction_visualization.png"
)

GRADCAM_PATH = (
    "results/plots/gradcam_visualization.png"
)

INFERENCE_PATH = (
    "results/plots/inference_example.png"
)