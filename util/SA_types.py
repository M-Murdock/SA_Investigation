from enum import Enum

class Inference(Enum):
    BAYESIAN = "Bayesian"
    MAX_ENT = "Max Entropy"
    CRF = "Conditional Random Fields"
    
class Assistance(Enum):
    DISTRIBUTION = "Distribution"

class Arbitration(Enum):
    LINEAR = "Linear"
    PROBABILISTIC = "Probabilistic"
    ONLY_USER = "User Action Only"