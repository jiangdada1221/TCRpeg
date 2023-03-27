# TCRpeg
TCRpeg is a deep probabilistic neural network framework used for inferring probability distribution for given CDR3 repertoires. Beyond that, TCRpeg can provide numerical embeddings for TCR sequences, generate new TCR sequences with highly similar statistical properties with the training repertoires. TCRpeg can be easily extended to act as a classifier for predictive purposes (TCRpeg-c). <br />

<img src="https://github.com/jiangdada1221/TCRpeg/blob/main/tcrpeg/figs/workflow_full.jpg" width="800"> <br />

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

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via [this link](https://drive.google.com/file/d/1rqgn6G2js85QS6K7mvMwOEepm4ARi54H/view?usp=sharing)

## Usage instructions

Define and train TCRpeg model:
```
from tcrpeg.TCRpeg import TCRpeg
model = TCRpeg(embedding_path='tcrpeg/data/embedding_32.txt',load_data=True, path_train=tcrs) 
#'embedding_32.txt' records the numerical embeddings for each AA; We provide it under the 'tcrpeg/data/' folder.
#'tcrs' is the TCR repertoire ([tcr1,tcr2,....])
model.create_model() #initialize the TCRpeg model
model.train_tcrpeg(epochs=20, batch_size= 32, lr=1e-3) 
```
Use the pretrained TCRpeg model for downstream applications:
```
log_probs = model.sampling_tcrpeg_batch(tcrs)   #probability inference
new_tcrs = model.generate_tcrpeg(num_to_gen=1000, batch_size= 100)    #generation
embs = model.get_embedding(tcrs)    #embeddings for tcrs
```

 We provide a tutorial jupyter notebook named [tutorial.ipynb](https://github.com/jiangdada1221/TCRpeg/blob/main/tutorial.ipynb). It contains most of the functional usages of TCRpeg which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation. <br />

 ## Command line usages

 We have provided the scripts for the experiments in the paper via the folder [tcrpeg/scripts](https://github.com/jiangdada1221/TCRpeg/tree/main/tcrpeg/scripts). <br />

 ```
python train.py --path_train ../data/TCRs_train.csv --epoch 20 --learning_rate 0.0001 --store_path ../results/model.pth 
```
To train a TCRpeg (with vj) model, the data file needs to have the columns named 'seq', 'v', 'j'. Insert 'python train.py --h' for more details.<br />
```
python evaluate.py --test_path ../data/pdf_test.csv --model_path ../results/model.pth
```
To compute the Pearson correlation coefficient of the probability inference task on test set. <br />
```
python generate.py --model_path ../results/model.pth --n 10000 --store_path ../results/gen_seq.txt
```
Use the pretrained TCRpeg to generate new sequences. Type 'python generate.py --h' for more details <br />
```
python classify.py --path_train ../data/train.csv --path_test ../data/test.csv --epoch 20 --learning_rate 0.0001
```
Use TCRpeg-c for classification task. The files should have two columns: 'seq' and 'label'. Type 'python classify.py --h' for more details. <br /> 
Note that the parameters unspecified will use the default ones (e.g. batch size) <br /><br />
The python files and their usages are shown below: <br />

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
```
Name: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. Otherwise, you may open an issue section in this repository.
```

## License

Free use of TCRpeg is granted under the terms of the GNU General Public License version 3 (GPLv3).

## Citation 
```
@article{jiang2023deep,
  title={Deep autoregressive generative models capture the intrinsics embedded in T-cell receptor repertoires},
  author={Jiang, Yuepeng and Li, Shuai Cheng},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={2},
  pages={bbad038},
  year={2023},
  publisher={Oxford University Press}
}
```
