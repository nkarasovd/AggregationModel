from time import time
from typing import Any, Callable


def timeit(func: Callable) -> Callable:
    def inner(*args, **kwargs) -> Any:
        start = time()
        print(f'\nStart method {func.__name__}')
        result = func(*args, **kwargs)
        print('Time to run:', time() - start)
        print('End')
        return result

    return inner
