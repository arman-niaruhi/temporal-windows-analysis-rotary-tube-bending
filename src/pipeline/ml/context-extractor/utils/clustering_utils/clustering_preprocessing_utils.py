from ..base_utils.base_preprocessor import BasePreprocessor

class ClusteringPreprocessor(BasePreprocessor):
    def __init__(self, sensors_path="../data/features.csv", target_path="../data/machine-and-movement.json"):
        super().__init__(sensors_path, target_path)
        # If JSON is used instead of CSV
        self.target_df = None