# TCRpeg
TCRpeg is a deep probabilistic neural network framework used for inferring probability distribution for given CDR3 repertoires. Beyond that, TCRpeg can provide numerical embeddings for TCR sequences, generate new TCR sequences with highly similar statistical properties with the training repertoires. TCRpeg can be easily extended to act as a classifier for predictive purposes (TCRpeg-c). <br />

<img src="https://github.com/jiangdada1221/tensorflow_in_practice/blob/master/workflow.png" width="500"> <br />

## Installation
TCRpeg is a python software implemented based on the deeplearning library - Pytorch. It is available on PyPI and can be downloaded and installed via pip:
 ```pip install tcrpeg```
TCRpeg can be also installed by cloning the Github repository and using the pip:
 ```pip install .```
The required software dependencies are listed below:
 ```
Numpy
matplotlib
tqdm
pandas
scikit-learn
scipy
torch >= 1.1.0
 ```

## Data

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via https:xxxxx

## Usage instructions

 We provide a tutorial jupyter notebook under the tcrpeg folder. It contains most functional usages of TCRpeg which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation.

## Contact

For instant enquiries, please contact us via [email](mailto:jiangdada12344321@gmail.com).

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).

