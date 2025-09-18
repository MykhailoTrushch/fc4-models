# FC4: : Fully Convolutional Color Constancy with Confidence-weighted Pooling

This repository provides an ONNX version of the FC4 model for estimation of the image illuminant, trained on the Gehler-Shi dataset.

---

## Original Resources

- **Original repository:** [https://github.com/yuanming-hu/fc4](https://github.com/yuanming-hu/fc4)
- **Paper:** [http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf)
- **PyTorch implementation:** [https://github.com/matteo-rizzo/fc4-pytorch/](https://github.com/matteo-rizzo/fc4-pytorch/)

---

## License

The original FC4 code and models, as well as the PyTorch implementation by Mateo Rizzo, are licensed under the **MIT License**.
See the [LICENSE](https://github.com/yuanming-hu/fc4/blob/master/LICENSE) file in the original repository, and [LICENSE](https://github.com/matteo-rizzo/fc4-pytorch/blob/main/LICENSE) file in the repository with the PyTorch implementation for details.

---

## Exporting FC4 to ONNX

The original paper on FC4 implements the algorithm in Python 2 and Tensorflow 1.x. Due to these versions being outdated, instead the PyTorch implementation of Mateo Rizzo was utilized.
This repository contains the fold 0 of the pretrained models available in the repository with the PyTorch implementation [LINK](https://github.com/matteo-rizzo/fc4-pytorch/releases/tag/1.0.2).

To export the ONNX version of FC4, we use the [test.py](https://github.com/matteo-rizzo/fc4-pytorch/blob/main/test/test.py) file with a pretrained model utilizing confidence-weighted pooling, taking fold 0 of the training result on the Gehler-Shi dataset.
The following is added to export the PyTorch model in the ONNX format:

```
torch.onnx.export(
    model._network,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=output_names,
    dynamic_axes=dynamic_axes
)
```

---

## Citation

Citation of the original paper:

```bibtex
@inproceedings{hu2017fc,
  title={FC 4: Fully Convolutional Color Constancy with Confidence-weighted Pooling},
  author={Hu, Yuanming and Wang, Baoyuan and Lin, Stephen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4085--4094},
  year={2017}
}
```
