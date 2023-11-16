class DivTargetBy:
    """Divide target by a factor.

    Args:
        factor (float): factor to divide by.
    """

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, x: float) -> float:
        """Divide target by a factor.

        Args:
            x (float): number to divide by factor

        Returns:
            float: divided number
        """
        return x / self.factor
