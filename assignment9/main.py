from collections import Counter
import heapq

# The image data provided in the question
image_data = [
    [1, 4, 2, 5, 4, 3, 3, 3],
    [4, 3, 0, 5, 2, 0, 4, 1],
    [2, 3, 1, 2, 1, 1, 3, 2],
    [5, 1, 1, 2, 2, 2, 2, 3],
    [1, 4, 1, 2, 0, 4, 3, 4],
    [1, 3, 1, 5, 1, 0, 1, 0],
    [1, 2, 2, 2, 5, 0, 4, 5],
    [1, 3, 0, 1, 5, 1, 1, 4]
]

# Flatten the matrix to a single list and count occurrences
flat_image_data = [num for row in image_data for num in row]
frequency_count = Counter(flat_image_data)

# Display the frequency for each number
print(frequency_count)

# Helper class for the nodes in the Huffman tree
class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    # Comparator function for the priority queue (min-heap)
    def __lt__(self, other):
        return self.frequency < other.frequency

# Build the Huffman Tree based on frequency count
def build_huffman_tree(frequencies):
    # Create a priority queue from the frequencies
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(priority_queue)

    # Merge nodes until one tree remains
    while len(priority_queue) > 1:
        # Pop the two nodes with lowest frequency
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        # Create a new node with these two nodes as children
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right

        # Add the new node back to the priority queue
        heapq.heappush(priority_queue, merged)

    # The remaining node is the root of the Huffman tree
    return priority_queue[0]

# Recursive function to generate Huffman codes
def generate_huffman_codes(node, code=""):
    if node is None:
        return {}

    # If it's a leaf node, return its symbol and code
    if node.symbol is not None:
        return {node.symbol: code}

    # Traverse the left and right subtree
    codes = {}
    codes.update(generate_huffman_codes(node.left, code + "0"))
    codes.update(generate_huffman_codes(node.right, code + "1"))
    return codes

# Build Huffman tree and generate codes
huffman_tree_root = build_huffman_tree(frequency_count)
huffman_codes = generate_huffman_codes(huffman_tree_root)

# Display the Huffman codes for each symbol
print(huffman_codes)

# Calculate the average number of bits (L_avg) based on the Huffman codes and frequencies
total_bits = sum(len(huffman_codes[symbol]) * freq for symbol, freq in frequency_count.items())
total_symbols = sum(frequency_count.values())

# Calculate the average number of bits per symbol
L_avg = total_bits / total_symbols
print(L_avg)

