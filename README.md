# check_GPU_experiments_PyTorch
Simple experiments for checking GPU availability, status, and speed using PyTorch

### Dependencies
- pytorch
- torchvision
- tqdm

### cuda_availability.py
- Checks CUDA availability and CUDA devices.
- Usage
```
    python pytorch_cuda_availability.py
```

### mnist.py
- Runs simple experiment with MNIST dataset.
- Usage
```
    python pytorch_mnist.py
        --batch_size [int]          (default: 128)
        --test_batch_size [int]     (default: 1000)
        --epochs [int]              (default: 10)
        --learning_rate [float]     (default: 0.1)
        --gpu_num [int]             (default: 0)
        --model [model name]        (default: 'cnn1')
        --no_cuda                   (default: False)
```
- Provided models: `cnn1`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
