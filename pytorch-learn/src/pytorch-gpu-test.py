import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

z = x @ y

print(z.device)
print(z.mean())
