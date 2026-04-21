import torchquery
print(f"Active Version: {torchquery.__version__}")

# Check if the new structure is there
try:
    from torchquery import TensorTricafig
    print("TensorTricafig is available!")
except ImportError:
    print("TensorTricafig not found - logic is likely inside QueryEngine.")