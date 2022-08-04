# Backdoor Attack on Federated-GAN
This is the PyTorch implementation of **[Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis](https://arxiv.org/abs/2207.00762)**.
## Abstract
Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.
## Usage
We are continue working on this project. Thus this repository contains all implementations in the paper and some other configurations beyond. Also, some configurations are hard-coded at this time, e.g. the malicious client index. Waiting to see our updated code.
### Environment
This project is based on PyTorch 1.10. You can simply set up the environment of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). 
We also provide 'environment.yml', but it may contain more libraries than needed for this project as we are progressing this project.

### Train
```bash
python scripts/vanilla.py -n_epochs 200 --batch 32 --attack --save_path ... --data_root ...

python scripts/vanilla.py -n_epochs 200 --batch 32 --attack --outlier_detect --save_path ... --data_root ...

python scripts/gp.py -n_epochs 200 --batch 32 --attack --save_path ... --data_root ...

python scripts/gp.py -n_epochs 200 --batch 32 --attack --outlier_detect --save_path ... --data_root ...
```

## Citation
If you find our project to be useful, please cite our paper.

```latex
@article{jin2022backdoor,
  title={Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis},
  author={Jin, Ruinan and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2207.00762},
  year={2022}
}
```

## Acknowledgements
Our coding and design are referred to the following open source repositories. Thanks to the greate people and their amazing work.

[stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

[projected_gan](https://github.com/autonomousvision/projected_gan)

[Hidden-Trigger-Backdoor-Attacks](https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks)