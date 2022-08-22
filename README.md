<a name="89366acb"></a>
## CNN based Lmser — Official PyTorch Implementation

---

This repository contains the official PyTorch implementation of the following paper:
> **Deep CNN Based Lmser and Strengths of Two Built-In Dualities**<br />[https://link.springer.com/article/10.1007%2Fs11063-020-10341-5](https://link.springer.com/article/10.1007%2Fs11063-020-10341-5)
> ​<br />
> **Abstract:** _Least mean square error reconstruction for the self-organizing network (Lmser) was proposed in 1991, featured by a bidirectional architecture with several built-in natures. In this paper, we developed Lmser into CNN based Lmser (CLmser), highlighted by new findings on strengths of two major built-in natures of Lmser, namely duality in connection weights (DCW) and duality in paired neurons (DPN). Shown by experimental results on several real benchmark datasets, DCW and DPN bring to us relative strengths in different aspects. While DCW and DPN can both improve the generalization ability of the reconstruction model on small-scale datasets and ease the gradient vanishing problem, DPN plays the main role. Meanwhile, DPN can form shortcut links from the encoder to the decoder to pass detailed information, so it can enhance the performance of image reconstruction and enables CLmser to outperform some recent state-of-the-art methods in image inpainting with irregularly and densely distributed point-shaped masks._

<a name="Resources"></a>
## Resources

---

Material related to our paper is available via the following links:

- Paper: [Deep CNN Based Lmser and Strengths of Two Built-In Dualities](https://www.cs.sjtu.edu.cn/~tushikui/publications/2020-Huang-NPL.pdf)
- Code: [https://github.com/CMACH508/CLmser](https://github.com/CMACH508/CLmser)


<br />Additional material can be found on Google Drive:

| Path | Description |
| --- | --- |
| [CLmser](https://drive.google.com/drive/folders/1hXtzE2HhJ7ywOs826PCs_ebppw662T7L?usp=sharing) | Main folder. |
| ├  [CLmser-paper.pdf](https://drive.google.com/file/d/1vh8_AIqpxraoMDk5_56-CfRrWG19T3aB/view?usp=sharing) | High-quality version of the paper PDF. |
| ├  [CelebA-HQ dataset](https://drive.google.com/file/d/1ggTJjGlI_7nKRH0c9Ur6lVMWlRY1z8pD/view?usp=sharing) | Preprocessed data for the [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans). |
| └  [Networks](https://drive.google.com/drive/folders/1u0iv4cJJvD3chdNtGA8kIO73Pg1KFWVU?usp=sharing) | Pre-trained networks on CelebA-HQ dataset. |
|    ├  [CLmser-celebahq.pth](https://drive.google.com/file/d/1Y45hngbzf9e0h9ESgVytDTZV-qT5Jp7a/view?usp=sharing) | CLmser trained with CelebA-HQ dataset at 256×256. |
|    ├  [CLmser_n-celebahq.pth](https://drive.google.com/file/d/10QzJAWPaet9e_p-8B66vV500DRzmfaoz/view?usp=sharing) | CLmser_n trained with CelebA-HQ dataset at 256×256. |
|    ├  [CLmser_w-celebahq.pth](https://drive.google.com/file/d/1HxugYvi8yiqow7tJHtgX7JGOz_KGUgTw/view?usp=sharing) | CLmser_w trained with CelebA-HQ dataset at 256×256. |
|    ├  [CAE-celebahq.pth](https://drive.google.com/file/d/1lfhsClIBBFA7ApzGAXk021aF5jiyUpm3/view?usp=sharing) | CAE trained with CelebA-HQ dataset at 256×256. |

<a name="1e496ca2"></a>
## Installation

---

Clone this repository:
```
git clone https://github.com/CMACH508/CLmser
cd Clmser
```
Install the dependencies:
```bash
conda create -n clmser python=3.6
conda activate clmser
conda install pytorch=0.4.1 cuda92 -c pytorch
pip install -r requirements.txt
```
We strongly recommend Linux for performance and compatibility reasons.
<a name="d4657c60"></a>
## Preparing datasets for training

---

1. To obtain the CelebA-HQ dataset (`data/celebahq`), please refer to the [Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans). The images are randomly split into 2,700/3,000 for train/set. We provide the resized and split CelebA-HQ dataset, you can get it from [resized-celebahq](n).
1. To obtain the Places2 dataset (`data/places2`), please refer to the [Places365-Standard](http://places2.csail.mit.edu/download.html). The train set in this implememntation contains 1/10  images  ( totally 180,000 images) of that in  [Places365-Standard](http://places2.csail.mit.edu/download.html). The test set contains 10,000 images randomly chosen from the original test set of [Places365-Standard](http://places2.csail.mit.edu/download.html). The lists of train and test sets  used in the paper are provided in [places2-lists](none).
<a name="636a371b"></a>
## Training networks

---

Once the datasets are set up, you can train the networks as follows:

1. Edit `configs/<DATASET>.json` to specify the dataset, model and training configuration.
1. Run the training script with `python train.py -c configs/<DATASET>.json --model_name <MODEL_NAME> --run_id <RUN_ID> `. For example, 
```bash
 # train CLmser with CelebA-HQ dataset on device GPU:0
python train.py -c configs/celebahq.json --model_name clmser --run_id 0 --device 0
```

3. The checkpoints are written to a newly created directory `saved/models/<MODEL_NAME>-<DATASET>/<RUN_ID>`
<a name="qoGDW"></a>
## Evaluation 

---

To reproduce the results for image inpainting, run the test script with `python train.py -c configs/<DATASET>.json --output_dir <OUTPUT_DIR> --model_name <MODEL_NAME> --resume <CHECKPOINT_PATH>`. For example, <br />

```bash
 # test CLmser with CelebA-HQ dataset
python test.py  -c configs/celebahq.json --output_dir results --model_name clmser  --resume pretrained/celebahq/clmser-celebahq.pth
```
​

Qualitative results are saved to `results/<MODEL_NAME-DATASET>-<RUN ID>`.<br />

<a name="sNSE0"></a>
## Citation

---

If you find this work useful for your research, please cite our paper:
```
@article{huang2020deep,
  title={Deep CNN Based Lmser and Strengths of Two Built-In Dualities},
  author={Huang, Wenjing and Tu, Shikui and Xu, Lei},
  journal={Neural Processing Letters},
  pages={1--17},
  year={2020},
  publisher={Springer}
}
```
## Acknowledgement

---
This repository used some codes in [pytorch-template](https://github.com/victoresque/pytorch-template).
<br />​<br />
