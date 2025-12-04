"""Sample prompts for different behavior categories.

These prompts are designed to elicit specific behaviors from the model
for interpretability analysis.
"""

# Factual recall prompts - test knowledge retrieval
FACTUAL_RECALL_PROMPTS = [
    "The capital of France is",
    "The Great Wall of China was built in",
    "Albert Einstein was born in the year",
    "Water boils at a temperature of",
    "The chemical symbol for gold is",
    "Mount Everest is located in",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The largest planet in our solar system is",
    "The currency of Japan is called",
]

# Reasoning prompts - test logical reasoning
REASONING_PROMPTS = [
    "If all cats are mammals and all mammals are animals, then all cats are",
    "If today is Tuesday, then tomorrow is",
    "2 + 3 = 5. 5 + 4 =",
    "If a train travels at 60 mph for 2 hours, it travels",
    "If A is greater than B and B is greater than C, then A is",
    "The next number in the sequence 2, 4, 6, 8 is",
    "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
    "If some doctors are tall and some tall people are athletes, can we conclude that some doctors are athletes? Answer:",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If you rearrange the letters 'CIFAIPC' you would have the name of a(n)",
]

# Code generation prompts - test programming ability
CODE_GENERATION_PROMPTS = [
    "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n    ",
    "# Function to reverse a string in Python\ndef reverse_string(s):\n    ",
    "// JavaScript function to check if a number is prime\nfunction isPrime(n) {\n    ",
    "# Python code to find the maximum element in a list\ndef find_max(lst):\n    ",
    "def factorial(n):\n    '''Calculate factorial recursively.'''\n    ",
    "# Binary search implementation\ndef binary_search(arr, target):\n    ",
    "// Function to calculate the sum of an array\nfunction sumArray(arr) {\n    ",
    "# Check if a string is a palindrome\ndef is_palindrome(s):\n    ",
    "def merge_sort(arr):\n    '''Implement merge sort algorithm.'''\n    ",
    "# FizzBuzz implementation\ndef fizzbuzz(n):\n    ",
]

# Multilingual prompts - test cross-lingual ability
MULTILINGUAL_PROMPTS = [
    "Translate to French: Hello, how are you?",
    "Translate to Spanish: The weather is nice today.",
    "Translate to German: I love programming.",
    "What does 'Bonjour' mean in English?",
    "Translate to Japanese: Thank you very much.",
    "Translate to Chinese: Good morning.",
    "What is 'Danke' in English?",
    "Translate to Italian: Where is the train station?",
    "Translate to Portuguese: I am learning a new language.",
    "What does 'Gracias' mean?",
]

# Mapping of behavior categories to their prompts
BEHAVIOR_PROMPTS = {
    "factual_recall": FACTUAL_RECALL_PROMPTS,
    "reasoning": REASONING_PROMPTS,
    "code_generation": CODE_GENERATION_PROMPTS,
    "multilingual": MULTILINGUAL_PROMPTS,
}


def get_prompts(behavior: str) -> list[str]:
    """Get prompts for a specific behavior category.

    Args:
        behavior: Name of the behavior category.

    Returns:
        List of prompts for the behavior.

    Raises:
        ValueError: If behavior is not recognized.
    """
    if behavior not in BEHAVIOR_PROMPTS:
        raise ValueError(
            f"Unknown behavior: {behavior}. "
            f"Available: {list(BEHAVIOR_PROMPTS.keys())}"
        )
    return BEHAVIOR_PROMPTS[behavior]


def get_all_prompts() -> dict[str, list[str]]:
    """Get all prompts organized by behavior.

    Returns:
        Dictionary mapping behavior names to prompt lists.
    """
    return dict(BEHAVIOR_PROMPTS)
