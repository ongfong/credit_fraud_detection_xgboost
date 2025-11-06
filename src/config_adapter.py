import json

class ConfigAdapter:
  
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_xgboost_params(self):
        tuned = self.config['tuned_params']
        fixed = self.config['fixed_params']
        
        xgb_params = {
            # Tuned params
            "num_round": tuned['n_estimators'],
            "eta": tuned['learning_rate'],
            "max_depth": tuned['max_depth'],
            "subsample": tuned['subsample'],
            "colsample_bytree": tuned['colsample_bytree'],
            
            # Fixed params
            "scale_pos_weight": fixed['scale_pos_weight'],
            "seed": fixed['random_state'],
            
            # XGBoost specific
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist"
        }
        
        return xgb_params
    
    def get_target_column(self):
        return self.config['target_column']
    
    def get_all(self):
        return self.config