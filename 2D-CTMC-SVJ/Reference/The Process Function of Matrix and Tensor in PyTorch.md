# The Process Function of Matrix and Tensor in PyTorch

**Author: Amber**

[TOC]

## 1. Conclusion

In recent years, researchers have created PyTorch to help them develop neural networks. This essay will show the difference between PyTorch and NumPy. After examining the correlation matrix and reviewing posts on CSDN and GITHUB, I conclude that if you want much faster speed, it's a very good choice to use PyTorch because it can utilize GPU to help you compute. And there isn't any very powerful matrix function unique to PyTorch because this is not what PyTorch focuses on. All the amazing matrix and tensor functions you think PyTorch has are based on your poor learning in NumPy. In this essay, I will list the basic functions of PyTorch for matrix and tensor operations. If you have more interest in this topic, you can go to CSDN for further study.

## 2. Function List

### 2.1 Basic Tensor Creation and Properties

PyTorch provides various ways to create tensors, which are the fundamental data structures for matrix and tensor operations.

#### 2.1.1 Tensor Creation Functions

```python
import torch

# Create tensors from data
data_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Data tensor: {data_tensor}")

# Create special tensors
zeros_tensor = torch.zeros(3, 4)  # 3x4 matrix of zeros
ones_tensor = torch.ones(2, 3)    # 2x3 matrix of ones
identity_matrix = torch.eye(3)    # 3x3 identity matrix
random_tensor = torch.randn(2, 4) # 2x4 random tensor from normal distribution

print(f"Zeros tensor shape: {zeros_tensor.shape}")
print(f"Identity matrix:\n{identity_matrix}")
```

**Explanation:** These functions create tensors with different initialization patterns. `torch.randn()` creates tensors with values sampled from a standard normal distribution, which is commonly used in machine learning applications.

#### 2.1.2 Tensor Properties

```python
# Create a sample tensor for property demonstration
sample_tensor = torch.randn(3, 4, 5)

# Basic properties
print(f"Shape: {sample_tensor.shape}")          # Tensor dimensions
print(f"Size: {sample_tensor.size()}")          # Same as shape
print(f"Number of dimensions: {sample_tensor.ndim}")
print(f"Total elements: {sample_tensor.numel()}")
print(f"Data type: {sample_tensor.dtype}")
print(f"Device: {sample_tensor.device}")
```

**Explanation:** These properties provide essential information about the tensor structure. Understanding tensor properties is crucial for debugging and ensuring operations are performed correctly.

### 2.2 Mathematical Operations

#### 2.2.1 Element-wise Operations

```python
# Create sample tensors
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)

# Basic arithmetic operations
addition = a + b  # or torch.add(a, b)
subtraction = a - b  # or torch.sub(a, b)
multiplication = a * b  # or torch.mul(a, b)
division = a / b  # or torch.div(a, b)

print(f"Addition:\n{addition}")
print(f"Element-wise multiplication:\n{multiplication}")

# Mathematical functions
sqrt_result = torch.sqrt(a)
exp_result = torch.exp(a)
log_result = torch.log(a)

print(f"Square root:\n{sqrt_result}")
print(f"Exponential:\n{exp_result}")
```

**Explanation:** Element-wise operations apply the operation to corresponding elements. These operations support broadcasting, allowing tensors of different but compatible shapes to be operated together.

#### 2.2.2 Matrix Operations

```python
# Matrix multiplication
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# Different ways to perform matrix multiplication
matmul_result = torch.matmul(a, b)  # General matrix multiplication
mm_result = torch.mm(a, b)          # 2D matrix multiplication
operator_result = a @ b             # Matrix multiplication operator

print(f"Matrix multiplication result shape: {matmul_result.shape}")

# Batch matrix multiplication
batch_a = torch.randn(10, 3, 4)  # 10 matrices of size 3x4
batch_b = torch.randn(10, 4, 5)  # 10 matrices of size 4x5
batch_result = torch.bmm(batch_a, batch_b)  # Result: 10x3x5

print(f"Batch multiplication result shape: {batch_result.shape}")
```

**Explanation:** Matrix multiplication is fundamental in linear algebra and neural networks. PyTorch provides different functions for different scenarios: `mm` for 2D matrices, `matmul` for general cases, and `bmm` for batch operations.

### 2.3 Statistical Functions

#### 2.3.1 Basic Statistics

```python
# Create sample data
data = torch.randn(4, 3)  # 4 samples, 3 features each

# Statistical operations
mean_all = torch.mean(data)           # Mean of all elements
mean_dim0 = torch.mean(data, dim=0)   # Mean along dimension 0 (across samples)
mean_dim1 = torch.mean(data, dim=1)   # Mean along dimension 1 (across features)

std_all = torch.std(data)             # Standard deviation
max_values, max_indices = torch.max(data, dim=1)  # Max values and indices
min_values, min_indices = torch.min(data, dim=1)  # Min values and indices

print(f"Overall mean: {mean_all}")
print(f"Mean per feature: {mean_dim0}")
print(f"Mean per sample: {mean_dim1}")
print(f"Max values per sample: {max_values}")
```

**Explanation:** Statistical functions are essential for data analysis and normalization. The `dim` parameter specifies which dimension to reduce, allowing for flexible aggregation patterns.

### 2.4 Shape Manipulation

#### 2.4.1 Reshaping Operations

```python
# Create a tensor to demonstrate reshaping
original = torch.arange(24)  # Numbers 0 to 23
print(f"Original shape: {original.shape}")

# Reshape operations
reshaped_2d = original.view(4, 6)      # Reshape to 4x6 matrix
reshaped_3d = original.view(2, 3, 4)   # Reshape to 2x3x4 tensor
reshaped_auto = original.view(-1, 8)   # Auto-calculate first dimension

print(f"2D reshape:\n{reshaped_2d}")
print(f"3D shape: {reshaped_3d.shape}")
print(f"Auto reshape shape: {reshaped_auto.shape}")

# Flatten operation
flattened = reshaped_3d.flatten()      # Flatten to 1D
flattened_partial = reshaped_3d.flatten(start_dim=1)  # Partial flatten

print(f"Flattened shape: {flattened.shape}")
print(f"Partially flattened shape: {flattened_partial.shape}")
```

**Explanation:** Shape manipulation is crucial for preparing data for different network layers. The `view()` operation creates a new view of the existing data without copying, while `flatten()` converts multi-dimensional tensors to lower dimensions.

#### 2.4.2 Dimension Operations

```python
# Dimension manipulation
tensor_3d = torch.randn(2, 3, 4)

# Add and remove dimensions
unsqueezed = tensor_3d.unsqueeze(1)    # Add dimension at position 1
squeezed = unsqueezed.squeeze(1)       # Remove dimension at position 1

# Transpose operations
transposed_2d = tensor_3d[0].t()       # Transpose 2D slice
permuted = tensor_3d.permute(2, 0, 1)  # Rearrange dimensions

print(f"Original shape: {tensor_3d.shape}")
print(f"After unsqueeze: {unsqueezed.shape}")
print(f"After squeeze: {squeezed.shape}")
print(f"After permute: {permuted.shape}")
```

**Explanation:** Dimension operations allow flexible manipulation of tensor shapes without changing the underlying data. These operations are essential for matching tensor shapes in complex neural network architectures.

### 2.5 Indexing and Slicing

#### 2.5.1 Basic Indexing

```python
# Create sample tensor
data = torch.randn(4, 5, 3)

# Basic indexing
first_sample = data[0]           # First sample: shape (5, 3)
specific_element = data[0, 2, 1] # Specific element
row_slice = data[:, 2, :]        # All samples, 3rd row, all columns

# Advanced indexing
indices = torch.tensor([0, 2, 3])
selected_samples = data[indices]  # Select specific samples

print(f"First sample shape: {first_sample.shape}")
print(f"Specific element: {specific_element}")
print(f"Selected samples shape: {selected_samples.shape}")
```

#### 2.5.2 Boolean Indexing and Masking

```python
# Boolean indexing
data = torch.randn(3, 4)
mask = data > 0  # Boolean mask for positive values

positive_values = data[mask]      # Extract positive values
data_masked = torch.where(mask, data, torch.zeros_like(data))  # Conditional replacement

print(f"Original data:\n{data}")
print(f"Positive values: {positive_values}")
print(f"Masked data:\n{data_masked}")

# Masked operations
data.masked_fill_(mask, -1)  # In-place operation
print(f"After masked fill:\n{data}")
```

**Explanation:** Boolean indexing and masking are powerful tools for conditional operations and data filtering, commonly used in data preprocessing and applying conditional logic.

### 2.6 Linear Algebra Operations

#### 2.6.1 Advanced Linear Algebra

```python
# Create sample matrices
A = torch.randn(4, 4)
B = torch.randn(4, 3)

# Matrix decompositions
U, S, V = torch.svd(A)           # Singular Value Decomposition
Q, R = torch.qr(A)               # QR Decomposition

# Matrix properties
det = torch.det(A)               # Determinant
trace = torch.trace(A)           # Trace
norm = torch.norm(A)             # Matrix norm

print(f"Determinant: {det}")
print(f"Trace: {trace}")
print(f"Norm: {norm}")

# Solving linear systems
try:
    inverse_A = torch.inverse(A)     # Matrix inverse
    solution = torch.mm(inverse_A, B)  # Solve Ax = B
    print(f"Solution shape: {solution.shape}")
except RuntimeError:
    print("Matrix is not invertible")
```

**Explanation:** These linear algebra operations are fundamental for many machine learning algorithms and numerical computations. PyTorch provides efficient implementations of these operations with GPU support.

### 2.7 GPU Acceleration Features

#### 2.7.1 GPU Operations

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    # Create tensors on GPU
    gpu_tensor = torch.randn(1000, 1000).to(device)
    gpu_tensor2 = torch.randn(1000, 1000, device=device)
    
    # Perform operations on GPU
    gpu_result = torch.mm(gpu_tensor, gpu_tensor2)
    
    # Move result back to CPU for printing
    cpu_result = gpu_result.cpu()
    
    print(f"GPU computation completed")
    print(f"Result shape: {cpu_result.shape}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
else:
    print("CUDA not available, using CPU")
```

**Explanation:** GPU acceleration is one of PyTorch's key advantages over NumPy. Operations can be seamlessly moved between CPU and GPU, providing significant speedup for large-scale computations.

### 2.8 Automatic Differentiation

#### 2.8.1 Gradient Computation

```python
# Create tensors that require gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=True)

# Define a computation
z = torch.sum(x**2 + y**3)  # z = x1² + x2² + y1³ + y2³

print(f"Forward computation result: {z}")

# Compute gradients
z.backward()

print(f"Gradient of x: {x.grad}")  # dz/dx = [2*x1, 2*x2]
print(f"Gradient of y: {y.grad}")  # dz/dy = [3*y1², 3*y2²]

# Verify manual calculation
manual_grad_x = 2 * x.data  # 2*[2.0, 3.0] = [4.0, 6.0]
manual_grad_y = 3 * y.data**2  # 3*[1.0², 4.0²] = [3.0, 48.0]

print(f"Manual gradient x: {manual_grad_x}")
print(f"Manual gradient y: {manual_grad_y}")
```

**Explanation:** Automatic differentiation is PyTorch's most powerful feature for machine learning. It automatically computes gradients of scalar outputs with respect to tensor inputs, enabling efficient training of neural networks.

### 2.9 Advanced Tensor Operations

#### 2.9.1 Einstein Summation

```python
# Einstein summation notation
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Traditional matrix multiplication
traditional = torch.mm(A, B)

# Einstein summation equivalent
einstein = torch.einsum('ik,kj->ij', A, B)

print(f"Traditional result shape: {traditional.shape}")
print(f"Einstein result shape: {einstein.shape}")
print(f"Results are equal: {torch.allclose(traditional, einstein)}")

# More complex example: batch matrix multiplication
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)

# Using einsum for batch operations
batch_result = torch.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"Batch einsum result shape: {batch_result.shape}")
```

#### 2.9.2 Tensor Stacking and Concatenation

```python
# Create sample tensors
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = torch.tensor([7, 8, 9])

# Stack tensors (creates new dimension)
stacked = torch.stack([t1, t2, t3], dim=0)  # Shape: (3, 3)
stacked_dim1 = torch.stack([t1, t2, t3], dim=1)  # Shape: (3, 3)

# Concatenate tensors (along existing dimension)
concatenated = torch.cat([t1, t2, t3], dim=0)  # Shape: (9,)

print(f"Stacked (dim=0):\n{stacked}")
print(f"Stacked (dim=1):\n{stacked_dim1}")
print(f"Concatenated: {concatenated}")

# Split operations
split_result = torch.split(concatenated, 3, dim=0)
chunk_result = torch.chunk(concatenated, 3, dim=0)

print(f"Split result lengths: {[t.shape for t in split_result]}")
```

**Explanation:** Stacking and concatenation operations are essential for combining multiple tensors, commonly used in batch processing and data pipeline construction.

