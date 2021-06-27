# HeCo
This repo is for source code of KDD 2021 paper "Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning". \
Paper Link: https://arxiv.org/abs/2105.09111
## Environment Settings
> python==3.8.5 \
> scipy==1.5.4 \
> torch==1.7.0 \
> numpy==1.19.2 \
> scikit_learn==0.24.2

GPU: GeForce RTX 2080 Ti \
CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
## Usage
Fisrt, go into ./code, and then you can use the following commend to run our model: 
> python main.py acm --gpu=0

Here, "acm" can be replaced by "dblp", "aminer" or "freebase".
## Some tips in parameters
1. We suggest you to carefully select the *“pos_num”* (existed in ./data/pos.py) to ensure the threshold of postives for every node. This is very important to final results. Of course, more effective way to select positives is welcome.
2. In ./code/utils/params.py, except "lr" and "patience", meticulously tuning dropout and tau is applaudable.
3. In our experiments, we only assign target type of nodes with original features, but assign other type of nodes with one-hot. This is because most of datasets used only provide features of target nodes in their original version. So, we believe in that if high-quality features of other type of nodes are provided, the overall results will improve a lot. The AMiner dataset is an example. In this dataset, there are not original features, so every type of nodes are all asigned with one-hot. In other words, every node has the same quality of features, and in this case, our HeCo is far ahead of other baselines. So, we strongly suggest that if you have high-quality features for other type of nodes, try it!
## Cite
## Contact
If you have any questions, please feel free to contact me with nianliu@bupt.edu.cn
