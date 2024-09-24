# A Vision Transformer-Based Framework for Knowledge Transfer From Multi-Modal to Mono-Modal Lymphoma Subtyping Models [IEEE-JBHI]

<details>
<summary>
    <b>A Vision Transformer-Based Framework for Knowledge Transfer From Multi-Modal to Mono-Modal Lymphoma Subtyping Models</b>. <a href="https://doi.org/10.1109/JBHI.2024.3407878" target="blank">[IEEE]</a> <a href="https://doi.org/10.48550/arXiv.2308.01328" target="blank">[arxiv]</a>
</summary>

```tex
@article{Guetarni2023AVT,
  title={A Vision Transformer-Based Framework for Knowledge Transfer From Multi-Modal to Mono-Modal Lymphoma Subtyping Models},
  author={Bilel Guetarni and F{\'e}ryal Windal and Halim Benhabiles and Marianne Petit and Romain Dubois and Emmanuelle Leteurtre and Dominique Collard},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  volume={28},
  pages={5562-5572},
}
```

**Abstract:** Determining lymphoma subtypes is a crucial step for better patient treatment targeting to potentially increase their survival chances. In this context, the existing gold standard diagnosis method, which relies on gene expression technology, is highly expensive and time-consuming, making it less accessibility. Although alternative diagnosis methods based on IHC (immunohistochemistry) technologies exist (recommended by the WHO), they still suffer from similar limitations and are less accurate. Whole Slide Image (WSI) analysis using deep learning models has shown promising potential for cancer diagnosis, that could offer cost-effective and faster alternatives to existing methods. In this work, we propose a vision transformer-based framework for distinguishing DLBCL (Diffuse Large B-Cell Lymphoma) cancer subtypes from high-resolution WSIs. To this end, we introduce a multi-modal architecture to train a classifier model from various WSI modalities. We then leverage this model through a knowledge distillation process to efficiently guide the learning of a mono-modal classifier. Our experimental study conducted on a lymphoma dataset of 157 patients shows the promising performance of our mono-modal classification model, outperforming six recent state-of-the-art methods. In addition, the power-law curve, estimated on our experimental data, suggests that with more training data from a reasonable number of additional patients, our model could achieve competitive diagnosis accuracy with IHC technologies. Furthermore, the efficiency of our framework is confirmed through an additional experimental study on an external breast cancer dataset (BCI dataset).

</details>

![overview](assets/jbhi-gagraphic-3407878.jpg)

## Installation

Please make sure the following packages are.

    - einops
    - numpy
    - opencv-python
    - openslide-python
    - pandas
    - pillow
    - scikit-learn
    - scikit-image
    - torch (1.8.1)
    - tqdm

The article expermients were done under Python 3.9 and PyTorch v1.8.1 with CUDA 10.1 on 4 Tesla V100 32Gb.

## Data Preprocess

Run `dataset.py` to extract sequences from WSI files and maunal annotations (must be QuPath exported annotations).
The annotation files must have the name as the WSI file associated and saved in directory a with stain name.

# Train multi and mono-modal models

To train the multi or mono-modal models run `train.py` while specifying the label CSV file containing the name of the slide with is class index.
To train the mlti-modal model use `--stain multi` argument, while for the mono-modal one `--stain mono`.

A very important detail: the argument `modalities` is used to specify which modalities to consider, and the first one is considered as the modality to keep after distillation.

# Test the models

Run `test.py` to test the mono-modal model.

## Reference

If you found our work useful in your research, please consider citing our works at:

```tex
@article{Guetarni2023AVT,
  title={A Vision Transformer-Based Framework for Knowledge Transfer From Multi-Modal to Mono-Modal Lymphoma Subtyping Models},
  author={Bilel Guetarni and F{\'e}ryal Windal and Halim Benhabiles and Marianne Petit and Romain Dubois and Emmanuelle Leteurtre and Dominique Collard},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  volume={28},
  pages={5562-5572},
}
```