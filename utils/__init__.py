# Utils package
from utils.data_loader import load_csv, load_all_datasets, validate_dataframe
from utils.analysis import compute_kpis, compute_trends, get_top_items, get_summary_statistics
from utils.helpers import format_currency, format_percentage, truncate_text
from utils.dataset_detector import DatasetDetector, detect_dataset, detect_all_datasets
