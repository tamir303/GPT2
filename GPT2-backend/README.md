# LLM Inference and Training API

This project provides a dual-purpose Python application that can either train a machine learning model (specifically a Language Learning Model, LLM) or serve inference requests via a FastAPI-based API. It leverages Hydra for configuration management during training and includes a robust set of FastAPI middlewares for secure, scalable inference.

## Features

- **Training Mode**: Trains an LLM using a configurable `TrainerPipeline` with parameters managed by Hydra.
- **Inference Mode**: Runs a FastAPI server with endpoints for model inference, secured with multiple middlewares.
- **Middlewares**:
  - Rate Limiting (100 requests/minute per client)
  - JWT-based Authentication
  - Cross-Origin Resource Sharing (CORS) Support
  - Custom Request/Response Logging
  - Global Error Handling
  - Prometheus Metrics for Monitoring
- **Default Behavior**: Runs in inference mode unless explicitly set to training mode via command-line argument.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning, if applicable)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone or Download the Project**:
   - Clone the repository:
     ```bash
     git clone https://github.com/tamir303/GPT2.git
     cd GPT2