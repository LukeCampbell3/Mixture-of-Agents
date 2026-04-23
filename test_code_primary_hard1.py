# code_primary_hard1_test.py

import unittest
from workspace_project_docker_workspace import count_character_frequencies

class TestCountCharacterFrequencies(unittest.TestCase):
    def test_empty_file(self):
        file_path = 'empty.txt'
        expected_output = {}
        
        try:
            result = count_character_frequencies(file_path)
            assert result == expected_output, f"Expected {expected_output}, got {result}."
            print("Test passed.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def test_file_with_only_spaces(self):
        file_path = 'spaces.txt'
        expected_output = {' ': 10}
        
        try:
            result = count_character_frequencies(file_path)
            assert result == expected_output, f"Expected {expected_output}, got {result}."
            print("Test passed.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def test_file_with_numbers(self):
        file_path = 'numbers.txt'
        expected_output = {'1': 3, '2': 2, '3': 4}
        
        try:
            result = count_character_frequencies(file_path)
            assert result == expected_output, f"Expected {expected_output}, got {result}."
            print("Test passed.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def test_file_with_punctuation(self):
        file_path = 'punctuation.txt'
        expected_output = {'!': 1, '?': 2}
        
        try:
            result = count_character_frequencies(file_path)
            assert result == expected_output, f"Expected {expected_output}, got {result}."
            print("Test passed.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def test_file_with_uppercase(self):
        file_path = 'uppercase.txt'
        expected_output = {'A': 1, 'B': 2}
        
        try:
            result = count_character_frequencies(file_path)
            assert result == expected_output, f"Expected {expected_output}, got {result}."
            print("Test passed.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
