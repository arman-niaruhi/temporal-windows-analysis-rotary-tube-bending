import pickle
import logging
from utils import take_all_bending_setups, multi_sensor_subplots

# --------------------------
# Setup logger
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_pipeline.log"),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Step 1: Load pickle file
# --------------------------
logger.info("Loading pickle file...")
with open('research/tubebend-main/experiments_process_and_results.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
logger.info(f"Loaded dictionary keys: {list(loaded_dict.keys())[:10]} ...")  # show first 10 keys

# --------------------------
# Step 2: Get all bending setups
# --------------------------
logger.info("Extracting all bending setups...")
all_bending_setups = take_all_bending_setups(loaded_dict)
logger.info(f"Total bending setups extracted: {len(all_bending_setups)}")

# --------------------------
# Step 3: Select a specific experiment
# --------------------------
experiment_number = 10
logger.info(f"Selecting experiment number {experiment_number}...")
experiment_as_dictinary = loaded_dict.get(f'Exp_{experiment_number}')
if experiment_as_dictinary is None:
    logger.error(f"Experiment Exp_{experiment_number} not found in loaded_dict")
    raise ValueError(f"Experiment Exp_{experiment_number} not found")
logger.info(f"Experiment sections: {list(experiment_as_dictinary.keys())}")

# --------------------------
# Step 4: Load process parameters, loads, and sensor data
# --------------------------
features_as_pandas_dataframe = experiment_as_dictinary['process_parameters_loads_sensor']
logger.info(f"DataFrame shape: {features_as_pandas_dataframe.shape}")
logger.info(f"Columns: {list(features_as_pandas_dataframe.columns)}")
logger.info(f"First 5 rows:\n{features_as_pandas_dataframe.head()}")

# --------------------------
# Step 5: Generate multi-sensor subplots
# --------------------------
logger.info("Generating multi-sensor subplots...")
#multi_sensor_subplots(features_as_pandas_dataframe, save_fig=True)

