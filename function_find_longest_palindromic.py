def longest_palindrome(s):
    if len(s) == 0:
        return ""

    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, end = 0, 0

    # All single characters are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start, end = i, i + 1

    # Check for lengths greater than 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start, end = i, j

    return s[start:end + 1]

# Time complexity: O(n^2)
# Space complexity: O(n^2)

# Test the function
s = "babad"
print(longest_palindrome(s))  # Output: "bab" or "aba"

# How it works:
# We use a 2D boolean array `dp` where dp[i][j] is true if the substring s[i:j+1] is a palindrome.
# We initialize all single characters as palindromes. Then, we check for substrings of length 2 and lengths greater than 2.
# For each substring, we use the information from the previous substring to determine if it is a palindrome.
