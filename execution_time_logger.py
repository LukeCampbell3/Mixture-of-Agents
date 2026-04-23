# execution_time_logger.py

import time
import functools

def execution_time_logger(max_retries=3, initial_delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
                    return result
                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        print(f"Retry failed: {e}")
                        time.sleep(initial_delay * (2 ** retries))
            raise Exception("Max retries exceeded")

        return wrapper

    return decorator

# Example usage
@execution_time_logger(max_retries=3, initial_delay=1)
def example_function(n):
    """Function to simulate an operation that might take a long time."""
    total = 0
    for i in range(n):
        total += i * i
    return total

result = example_function(1000000)
print(result)
