# GPU-vRAM Usage Estimation for Diffusion Models

## Objective
Derive an analytical equation to estimate peak vRAM usage during inference for the `stable-diffusion-v1-5/stable-diffusion-v1-5` for arbitrary input image sizes.

## Background
vRAM consumption during diffusion model inference differs significantly from model size on disk. Peak memory depends on:
 - Model weights (fixed)
 - Intermediate activations (vary with image dimensions and prompt length)
 - Framework overhead (CUDA kernels, workspace buffers)
 - Attention mechanism memory scaling (O(N²) with sequence length)

Where:
 - `H`, `W` = input image height and width
 - `prompt_length` = tokenized prompt length
 - Identify any additional factors affecting vRAM

## Requirements
 - Analyze the architecture: Understand UNet, VAE, CLIP text encoder, and how tensors flow through the pipeline
 - Account for precision: Assume `FP16` (2 bytes/parameter)
 - Model fully on GPU: Ignore pipeline.enable_model_cpu_offload() in your equation
 - Peak, not average: Find the stage with maximum memory allocation
 - Document assumptions: Clearly state what you include/exclude (e.g., gradient storage, optimizer states)

## Deliverables
 - Equation with explanation of each term
 - Derivation notes showing how you arrived at each component
 - Validation (optional but encouraged): Compare equation predictions against actual nvidia-smi measurements using the provided test code

## Your Task
Derive a formula for vRAM usage.

## Tips
- Although no GPU is needed to accomplish this task (analyze code/architecture)
- Use PyTorch documentation and model architecture inspection

## Evaluation Criteria
- Correctness: Formula accounts for major memory consumers
- Completeness: All image-dependent and prompt-dependent factors identified
- Rigor: Derivation shows understanding of PyTorch memory model and diffusion architecture
- Clarity: Equation is readable and well-documented
