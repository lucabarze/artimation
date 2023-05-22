# Artimation

## Introduction
Artimation is a real-time, AI-powered webcam art creation application. With the power of your webcam and your fingers, you can draw on a virtual canvas in the application. Artimation leverages TensorFlow and MediaPipe to recognize hand movements and generate beautiful, stylized art.

## Features
- Real-time video feed from your webcam.
- Drawing on the screen with your index finger.
- Reset the canvas with a simple fist gesture.
- The lines you draw are processed with a generative art effect.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker installed 
- A Kubernetes cluster (like microk8s or minikube) 
- Helm v3 installed 
- Akri installed in the Kubernetes cluster

### Installing

Clone the repository:

```bash
git clone https://github.com/lucabarze/artimation.git
cd artimation
```

Build the Docker image:

```bash
docker build -t your-dockerhub-username/artimation:latest .
```

Push the image to DockerHub:

```bash
docker push your-dockerhub-username/artimation:latest
```

Deploy the application in your Kubernetes cluster using Helm:

```bash
kubectl apply -f kubernetes_resources/*
```

### Akri Integration

Artimation integrates with Akri to handle the dynamic discovery and management of the webcam as a leaf device. Akri's custom resources, Discovery Handlers, Agents, and Controller are utilized to provide an abstraction layer that simplifies the process of finding, utilizing, and monitoring the availability of the webcam.

The files within the `kubernetes_resources` folder in this repository are Akri resource definitions and are used by Artimation to enable the integration with Akri.

## Built With

- Flask
- TensorFlow
- MediaPipe
- Docker
- Kubernetes
- Akri

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This project wouldn't be possible without the incredible work done by the Akri team. We're grateful for their contributions to the cloud native and edge computing communities.
