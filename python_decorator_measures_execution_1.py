from retry_decorator import retry_decorator

@retry_decorator(max_retries=3, initial_delay=1.0)
def example_function(arg):
    """
    Example function that takes an argument and returns its square.
    """
    time.sleep(2)  # Simulate some work
    return arg ** 2

try:
    result = example_function(5)
    print(f"Result: {result}")
except Exception as e:
    print(f"An error occurred: {e}")
