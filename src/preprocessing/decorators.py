import time

from typing import Callable


def transformer_time_measurement_decorator(transformer_name: str) -> Callable:
    """Decorator for measuring duration of running pre-processing transformer.

    Note: This code is taken and modified from one of my previous projects:
        https://github.com/pmacinec/diabetes-patients-readmissions-prediction

    Args:
        transformer_name: Name of the pre-processing transformer (e.g. StopwordsFilter).

    Returns:
        Callable: Decorator function to measure time of transformer.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = function(*args, **kwargs)

            end_time = time.time()
            duration = round(end_time - start_time, 2)
            print(f'  >> {transformer_name} transformation ended, took {duration} seconds.')

            return result
        return wrapper
    return decorator
