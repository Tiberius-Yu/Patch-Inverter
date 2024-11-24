## Patch Inverter: A Novel Block-wise GAN Inversion Method for Arbitrary Image Resolutions

Official implementation for paper: Patch Inverter: A Novel Block-wise GAN Inversion Method for Arbitrary Image Resolutions

> **Abstract** Generative adversarial networks (GANs) have achieved remarkable progress in generating realistic images from merely small dimensions, which essentially establishes the latent generating space by rich semantics. GAN inversion thus aims at mapping real-world images back into the latent space, allowing for the access of semantics from images. However, existing GAN inversion methods can only invert images with fixed resolutions; this significantly restricts the representation capability in real-world scenarios. To address this issue, we propose to invert images by patches, thus named as patch inverter, which is the first attempt in terms of block-wise inversion for arbitrary resolutions. More specifically, we develop the padding-free operation to ensure the continuity across patches, and analyse the intrinsic mismatch within the inversion procedure. To relieve the mismatch, we propose a shifted convolution operation, which retains the continuity across image patches and simultaneously enlarges the receptive field for each convolution layer. We further propose the reciprocal loss to regularize the inverted latent codes to reside on the original latent generating space, such that the rich semantics can be maximally preserved. Experimental results have demonstrated that our patch inverter is able to accurately invert images with arbitrary resolutions, whilst representing precise and rich image semantics in real-world scenarios.


## Requirements

### Dependencies

- Python 3.6
- PyTorch 1.8
- Opencv

You can install a new environment for this repo by running
```
conda env create -f environment.yml
conda activate PatchInv
pip install tensorboardx==2.1 easydict python-lmdb tqdm
```


## Training

* Prepare the training data

    Preparing your dataset or using dataset provided according to [InfinityGAN](https://github.com/hubert0527/infinityGAN)
    
* Training

    You can modify the training options of the config file in the directory `configs/` and the path in `train-sample.sh`.
    
    ```
    bash train-sample.sh
    ```

## Testing 

* Inversion

    ```
    bash infer-sample.sh
    ```

# TODOs

- [ ] refine released train code
- [ ] refine test code
- [ ] exhibit the feature edit method
- [ ] release ckpt

## Citation

```
@article{
}
```
## License

Copyright © 2024, All rights reserved.

This source code is made available under the license found in the LICENSE.txt in the root directory of this source tree.



