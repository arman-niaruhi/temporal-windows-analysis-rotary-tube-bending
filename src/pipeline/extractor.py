import pickle
from src.logging.log_utils import log_function
from src.utils.utils import take_all_bending_setups


class data_extractor():
    def __init__(self) -> None:
        with open("data/experiments_process_and_results.pkl", "rb") as f:
            self.loaded_dict = pickle.load(f)
         
    @log_function   
    def load_dict_getter(self):
        """Getter for self.load_dict
            obj.load_dict_getter """  
        return self.loaded_dict
    
    @log_function
    def get_all_bending_setups(self, machine_part):
        """Return all bending setups from the loaded dictionary."""
        return take_all_bending_setups(self.loaded_dict, machine_part)

    @log_function
    def get_experiment_by_number(self, experiment_number):
        """Return a specific experiment as a dictionary given its number."""
        return self.loaded_dict.get(f'Exp_{experiment_number}')

    @log_function
    def get_experiment_keys(self):
        """Print keys of an experiment dictionary with numbering."""
        for key in list(self.loaded_dict.get(f'Exp_{2}')):
            print(key)
        