# code_primary_hard1.py

def count_character_frequencies(file_path):
    """
    Read a text file and return the frequency of each character in it.
    
    Parameters:
        file_path (str): The path to the input text file.
        
    Returns:
        dict: A dictionary with characters as keys and their frequencies as values.
    """
    if not isinstance(file_path, str):
        raise ValueError("The file_path must be a string.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    
    char_frequencies = {}
    for char in content:
        if char.isalpha():  # Only count alphabetic characters
            char = char.lower()  # Convert to lowercase to handle case insensitivity
            char_frequencies[char] = char_frequencies.get(char, 0) + 1
    
    return char_frequencies

# Test the function
def test_count_character_frequencies():
    file_path = 'example.txt'  # Replace with a valid text file path
    expected_output = {'a': 3, 'b': 2, 'c': 4}
    
    try:
        result = count_character_frequencies(file_path)
        assert result == expected_output, f"Expected {expected_output}, got {result}."
        print("Test passed.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_count_character_frequencies()
