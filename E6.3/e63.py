import random
import string
import zlib

# Generate a long random string
random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10000))

# Compress the string using zlib compression
compressed_data = zlib.compress(random_string.encode())

# Calculate compression ratio
compression_ratio = len(random_string) / len(compressed_data)

print("Original string length:", len(random_string))
print("Compressed data length:", len(compressed_data))
print("Compression ratio:", compression_ratio)

print(
    "The expected compression ratio for a random string generated in this manner is likely to be close to 1 or slightly above 1. This is because random data, such as a randomly generated string, does not have much redundancy or patterns that can be effectively compressed by lossless compression algorithms. In fact, random data can sometimes even result in larger compressed sizes due to the overhead introduced by the compression algorithm."
)
