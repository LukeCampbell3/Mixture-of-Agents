def retry(exceptions, max_retries=3, initial_delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    logger.info(f"Function '{func.__name__}' executed successfully in {end_time - start_time:.4f} seconds.")
                    return result
                except exceptions as e:
                    logger.error(f"Exception occurred in '{func.__name__}'. Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(initial_delay * (2 ** attempt))
                    attempt += 1
            logger.error(f"All retries failed for '{func.__name__}' after {max_retries} attempts.")
        return wrapper
    return decorator
