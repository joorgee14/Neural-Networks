# Neural-Networks

Project Grades : 10

Kaggle Competition Final Position in Private Leaderboard --> 2nd

AUC score -> 0.94327

This coursework focused on:

- Developing solutions using Neural Networks with PyTorch

- Gaining deep knowledge of deep learning and various network architectures

- Solving tasks using MLPs, CNNs, and RNNs

- Learning about Transformers, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models

The course included four projects, two partial exams, a Kaggle competition, and a final exam.

- Project 1: Autoencoder
Designed and trained Autoencoders on the MNIST and FMNIST datasets, exploring different architectures and regularization techniques, and analyzing their effects.

- Project 2: Network Calibration
Based on the paper "On Calibration of Modern Neural Networks", trained a LeNet-5 model on a modified CIFAR-10 dataset, studied calibration curves, implemented Temperature Scaling, and optionally tested with deeper networks.

- Project 3: Attention Mechanism
Implemented an attention mechanism from scratch for an LSTM-based RNN, following specific design instructions, and compared its performance against a standard LSTM without attention.

- Project 4: Variational Autoencoder (VAE)
Completed the implementation of a VAE for the CelebA dataset using convolutional networks. Additionally, built a VAE for synthetic data from a 3D Gaussian Mixture Model using dense layers. Compared generated samples to the ground truth, evaluated missing modes, and used T-SNE to visualize and analyze clustering behavior in the latent space.

Kaggle Competition:
This competition focused on binary classification of skin lesions using the ISIC (International Skin Imaging Collaboration) dataset. The task was to build a robust model capable of distinguishing malignant from benign skin lesions from dermoscopic images.

Participants were provided with:

A training set of labeled images (target = 1 for malignant, 0 for benign)

A separate test set of unlabeled images

The goal was to predict the probability that each test image was malignant

The main challenge lay in the class imbalance (malignant cases were the minority), requiring careful preprocessing, model tuning, and evaluation strategies. The final metric used to rank submissions was the Area Under the ROC Curve (AUC).

We approached the task using deep learning (CNNs) with EfficientNet backbones, applied cross-validation, handled imbalance using sampling techniques, and visualized model decisions using Grad-CAM and saliency maps.
