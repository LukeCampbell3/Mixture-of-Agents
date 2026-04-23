import time
import functools
import random

def retry(exceptions=(Exception,), max_retries=3, initial_delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    # Measure execution time
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
                    return result
                except exceptions as e:
                    attempt += 1
                    # Exponential backoff delay
                    delay = initial_delay * (2 ** attempt)
                    print(f"Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(delay)
            raise Exception("Max retries exceeded")
        return wrapper
    return decorator

@retry(exceptions=(Exception,), max_retries=3, initial_delay=1)
def slow_function(n):
    # Simulate a slow operation
    time.sleep(random.uniform(0.5, 2))
    print(f"Slow function executed with n={n}")
    return n * 2

if __name__ == "__main__":
    slow_function(100)
