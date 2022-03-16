# TCRpeg
TCRpeg is a deep probabilistic neural network framework used for inferring probability distribution for given CDR3 repertoires. Beyond that, TCRpeg can provide numerical embeddings for TCR sequences, generate new TCR sequences with highly similar statistical properties with the training repertoires. TCRpeg can be easily extended to act as a classifier for predictive purposes (TCRpeg-c). <br />

<img src="https://github.com/jiangdada1221/tensorflow_in_practice/blob/master/workflow.png" width="800"> <br />

## Installation
TCRpeg is a python software implemented based on the deeplearning library - Pytorch. It is available on PyPI and can be downloaded and installed via pip: <br />
 ```pip install tcrpeg``` <br />
TCRpeg can be also installed by cloning the Github repository and using the pip: <br />
 ```pip install .``` <br />
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

 We provide a tutorial jupyter notebook named tutorial.ipynb. It contains most of the functional usages of TCRpeg which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation. The python scripts and their usages are shown below: <br />

| Module name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| TCRpeg.py                                      | Contain most functions of TCRpeg                   |
| evaluate.py                                    | Evaluate the performance of probability inference  |
| word2vec.py                                    | word2vec model for obtaining embeddings of AAs     |
| model.py                                       | Deep learning models of TCRpeg,TCRpeg-c,TCRpeg_vj  |
| classification.py                              | Apply TCRpeg-c for classification tasks            |
| utils.py                                       | N/A (contains util functions)                      |
| process_data.py                                | Construct the universal TCR pool                   |

## Contact

We check email often, so for instant enquiries, please contact us via [email](mailto:jiangdada12344321@gmail.com). Or you may open an issue section.

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).

