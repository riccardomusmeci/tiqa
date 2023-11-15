
class DivTargetBy:
    def __init__(self, factor: float) -> None:
        self.factor = factor
        
    def __call__(self, x: float) -> float:
        return x / self.factor