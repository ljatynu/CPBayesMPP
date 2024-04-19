# Contrastive Prior Enhances the Performance of Bayesian Neural Network-based Molecular Property Prediction (CPBayesMPP)

This repository is an implementation of our paper "Contrastive Prior Enhances the Performance of Bayesian Neural Network-based Molecular Property Prediction" in PyTorch. In this work, we propose a method called **<u>CPBayesMPP</u>**, aiming at enhancing (1) prediction accuracy and (2) uncertainty quantification capacity (UQ) performance of Bayesian Neural Network (BNN).

The overview of CPBayesMPP is shown as following:

![Alts](figures/The%20framework%20of%20proposed%20CPBayesMPP/The%20framework%20of%20proposed%20CPBayesMPP.JPG)

CPBayesMPP contains two main stages:

- **Stage (a): Learn contrastive prior from unlabeled dataset.** Instead of traditionally specifying an uninformative prior (e.g., isotropic Gaussian) on the parameters, we try to learn an informative contrastive prior from the large-scale unlabeled dataset, which can more precisely describe the molecular structural space.
- **Stage (b): Infer enhanced task-specific posterior from labeled dataset.** Here, incorporating the learned contrastive prior, we infer a prior-enhanced task-specific posterior for the parameters, which can improve the prediction accuracy and UQ performance for the downstream datasets.

The details of CPBayesMPP are described in our paper.

## Table of contents

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Stage (a): Pre-training to learn contrastive prior](#stage-a-pre-training-to-learn-contrastive-prior)
- [Stage (b): Infer enhanced task-specific posterior](#stage-b-infer-enhanced-task-specific-posterior)
- [Stage (c): Enhanced-posterior prediction case studies](#stage-c-enhanced-posterior-prediction-case-studies)
    - [Case study 1. Prediction accuracy Improvement](#case-study-1-prediction-accuracy-improvement)
    - [Case study 2. Uncertainty quantification performance improvement](#case-study-2-uncertainty-quantification-performance-improvement)
    - [Case study 3. Active learning performance improvement](#case-study-3-active-learning-performance-improvement)
    - [Case study 4. Prior Prediction performance of contrastive prior](#case-study-4-prior-prediction-performance-of-contrastive-prior)
- [Enhance the task-specific posterior on customized dataset](#enhance-the-task-specific-posterior-on-customized-dataset)
- [Thanks](#thanks)
- [Cite us](#cite-us)

## Requirements

- torch 1.8.2+cu102
- numpy 1.24.4
- scikit-learn 1.10.1
- rdkit 2022.9.5

## Datasets

- **Pre-training dataset:** We randomly extract 1 million unlabeled molecules from <u>[ChemBERTa](https://arxiv.org/abs/2010.09885)</u> to form the contrastive pre-training dataset and save them in the `pubchem-1m-clean.csv` file. You can download it <u>[here](https://drive.google.com/file/d/1FO_otxK3WHA629Xu1oseDWJPPJCUbgu5/view?usp=sharing)</u> and place it in the `/dataset` folder.
- **Downstream dataset:** we conduct experiments on 6 regression and 6 classification downstream datasets, including:
    - **Regression benchmarks:** ESOL (named as Delaney), FreeSolv, Lipo, QM7, QM8 and PDBbind.
    - **Classification benchmarks:** BACE, Tox21, HIV, BBBP, SIDER and ClinTox.
    - All these datasets have been saved in the `/dataset` folder under the corresponding names. You can also find the original ones from <u>[MoleculeNet](https://moleculenet.org/datasets-1)</u>.

## Stage (a): Pre-training to learn contrastive prior

Run the script **[0-pretrain.py](0-pretrain.py)** by the following command for contrastive prior learning. (For the detailed description, please refer to **Section 3.3 Learn contrastive prior from unlabeled dataset** in the paper)

```bash 
python 0-pretrain.py --batch_size 512 --epochs 50 --save_dir results/pretrain --pretrain_data_name pubchem-1m-clean 
```

The pre-training MPNN encoder will be saved as `/results/pretrain/pretrain_encoder.pt`, while the transformation header will be saved as `/results/pretrain/pretrain_header.pt`.

**Note: How to accelerate pre-training?**

We recommend using a _cache mechanism_ to speed up the pre-training process as follows, because we have found in practice that the main overhead of the pretraining process comes from the augmentation operations on contrastive samples.

* Step 1: Run the script [prepare_pretrain_cache.py](dataset/prepare_pretrain_cache.py) to generate and save the augmented sample pairs in advance. This will save the batch augmented samples, with the name `smiles_to_contra_graph_batch_0.pkl`, to the `/pubchem-1m-clean-cache` folder. You can download the preprocessed cache file [here](https://drive.google.com/file/d/1w6fz1X1IF00UigEjmzh6UbTOvrqKhtPk/view?usp=sharing) and extract it to the /dataset folder.

* Step 2: Run the following command to use cache for accelerating the pretraining process (on an 11GB NVIDIA GeForce 2080Ti GPU, each epoch takes ~ 30 minutes).

```bash
python 0-pretrain.py --batch_size 512 --epochs 50 --save_dir results/pretrain --pretrain_data_name pubchem-1m-clean --use_pretrain_data_cache True
```

## Stage (b): Infer enhanced task-specific posterior

**CPBayesMPP:** In downstream tasks, you can use the following commands to infer the enhanced task-specific posterior (Remember to move the pre-trained folder `results/pretrain` into the folder `results/cl/pretrain`).

```bash
python 1-train.py --data_name delaney --train_strategy cl --epoch 200 --split_type random --split_sizes 0.5 0.2 0.3
```

This will output the trained model to the `results/cl/random_split/delaney_checkpoints` folder, and save the variational parameters under the corresponding `seed_i/model` for each random seed i.

**BayesMPP:** For comparison, you can run the following command, which will train the model using an uninformative prior.

```bash
python 1-train.py --data_name delaney --train_strategy cd --epoch 200 --split_type random --split_sizes 0.5 0.2 0.3
```

**Notes:**

- The `--train_strategy` parameter can be set to `cl` or `cd` for training, respectively. `cl` means contrastive learning with contrastive prior, and `cd` means concrete dropout with uninformative prior.
- In regression tasks, we use `random split` with a `50/20/30` ratio, while in classification tasks, we use `scaffold split` with a `80/10/10` ratio.
- See more details in the section **4.4 Model Training Detail** in the paper.

## Stage (c): Enhanced-posterior prediction case studies

**You can reproduce all experimental results in our paper through the following case studies.**

<hr style="border: 0.1px">

### Case study 1. Prediction accuracy Improvement

**Step (1):** Evaluate the prediction performance of the trained model with contrastive prior and uninformative prior, respectively.
(RMSE for regression tasks and AUC-ROC for classification tasks)

```bash 
python 2-1-predict.py --data_name delaney --train_strategy cl --split_type random
```

```bash 
python 2-1-predict.py --data_name delaney --train_strategy cd --split_type random
```

The prediction performance results will be saved as `delaney_metric.csv` in the corresponding folders.

**Step (2):** Calculate the statistics of the prediction performance improvement (mean and std under 8 random seeds).

```bash
python 2-2-visualize_predict_performance.py --data_name delaney --train_strategy cl
```

```bash
python 2-2-visualize_predict_performance.py --data_name delaney --train_strategy cd
```

**Notes:** See more details in the section **5.1 Predictive performance improvement** in the paper.

<details>
  <summary><b>Click here for the results!</b></summary>

<b>Performance comparison (RMSE, the lower the better) of different methods on 6 regression datasets</b>

| ![Alts](figures/RMSE%20comparison%20on%206%20regression%20datasets/RMSE%20comparison%20on%206%20regression%20datasets.JPG) |
|----------------------------------------------------------------------------------------------------------------------------|

<b>Performance comparison (AUC-ROC, the higher the better) of different methods on 6 classification datasets</b>

| ![Alts](figures/AUC-ROC%20comparison%20on%206%20classification%20datasets/AUC-ROC%20comparison%20on%206%20classification%20datasets.JPG) |
|------------------------------------------------------------------------------------------------------------------------------------------|

</details>

<hr style="border: 0.1px">

### Case study 2. Uncertainty quantification performance improvement

**Step (1):** Evaluate the prediction uncertainty of the model trained with contrastive prior and uninformative prior, respectively.

```bash
python 3-1-uncertainty_predict.py --data_name delaney --train_strategy cl --split_type random
```

```bash
python 3-1-uncertainty_predict.py --data_name delaney --train_strategy cd --split_type random
```

**Step (2):** Plot the uncertainty calibration curves.

```bash
python 3-2-visualize_uncertainty.py --data_name delaney --split_type random --visualize_type auco --uncertainty_type aleatoric
```

**Notes:**

* Specify `--uncertainty_type` as `aleatoric`, `epistemic` or `total` to visualize the different uncertainty calibration curves.
* Set `--visualize_type auco` and `--split_type random` for regression tasks while `--visualize_type ece` and `--split_type scaffold` for classification ones.
* Regression uncertainty curves will be saved in the folder `/figures/Uncertainty calibration curves for regression datasets` while classification ones will be saved in the folder `/figures/Uncertainty calibration curves for classification datasets`.
* See more details in the section **5.2 Uncertainty quantification performance improvement** in the paper.

<details>
    <summary><b>Click here for the results!</b></summary>

<b>Uncertainty Calibration Curves for ELSO dataset</b>

| ![Alts](figures/Uncertainty%20calibration%20curves%20for%20regression%20datasets/ESOL%20Aleatoric%20Uncertainty.JPG) | ![Alts](figures/Uncertainty%20calibration%20curves%20for%20regression%20datasets/ESOL%20Epistemic%20Uncertainty.JPG) | ![Alts](figures/Uncertainty%20calibration%20curves%20for%20regression%20datasets/ESOL%20Total%20Uncertainty.JPG) |
|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|

<b>AUCO Performance improvement on regression datasets:</b>

| ![Alts](figures/AUCO%20comparision%20on%206%20regression%20datasets/AUCO%20comparision%20on%206%20regression%20datasets.JPG) |
|------------------------------------------------------------------------------------------------------------------------------|

<b>Uncertainty Calibration Curves for Classification dataset</b>

| ![Alts](figures/Uncertainty%20calibration%20curves%20for%20classification%20datasets/BACE.JPG) | ![Alts](figures/Uncertainty%20calibration%20curves%20for%20classification%20datasets/HIV.JPG) | ![Alts](figures/Uncertainty%20calibration%20curves%20for%20classification%20datasets/BBBP.JPG) |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|

<b>ECE Performance improvement on classification datasets</b>

| ![Alts](figures/ECE%20comparision%20on%203%20classification%20datasets/ECE%20comparision%20on%203%20classification%20datasets.JPG) |
|------------------------------------------------------------------------------------------------------------------------------------|

</details>

<hr style="border: 0.1px">

### Case study 3. Active learning performance improvement

**Step (1):** Train the model by 4 strategies:

* ① BayesMPP + Random Learning
* ② BayesMPP + Active Learning
* ③ CPBayesMPP + Random Learning
* ④ CPBayesMPP + Active Learning

```bash
python 4-1-active_train.py --data_name freesolv --train_strategy cd --active_learning_type random --init_train_step 100 --active_train_step 30
```

```bash
python 4-1-active_train.py --data_name freesolv --train_strategy cd --active_learning_type explorative --init_train_step 100 --active_train_step 200
```

```bash
python 4-1-active_train.py --data_name freesolv --train_strategy cd --active_learning_type random --init_train_step 200 --active_train_step 30
```

```bash
python 4-1-active_train.py --data_name freesolv --train_strategy cl --active_learning_type explorative --init_train_step 150 --active_train_step 300
```

**Step (2):** Visualize the active learning curves.

```bash
python 4-2-visualize_active.py --data_name freesolv
```

**Notes:**

* For fairness, due to the difficulty of gradient descent on explorative samples, we appropriately increase the number of gradient descent iterations for Active Learning strategy.
* See more details in the section **5.3 Active learning performance improvement** in the paper.

<details>
  <summary><b>Click here for the results!</b></summary>

<b>Performance changes in Active Learning</b>

| ![Alts](figures/Performance%20changes%20in%20Active%20Learning/ESOL%20(Regression%20dataset).JPG) | ![Alts](figures/Performance%20changes%20in%20Active%20Learning/FreeSolv%20(Regression%20dataset).JPG) | ![Alts](figures/Performance%20changes%20in%20Active%20Learning/BACE%20(Classification%20dataset).JPG) | ![Alts](figures/Performance%20changes%20in%20Active%20Learning/BBBP%20(Classification%20dataset).JPG) |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|

</details>

<hr style="border: 0.1px">

### Case study 4. Prior Prediction performance of contrastive prior

**Step (1):** Predict Feature similarity using contrastive prior and uninformative prior, respectively.

```bash
python 5-1-prior_predict.py --data_name delaney --train_strategy cl --prior_predict_type prior_similarity
```

```bash
python 5-1-prior_predict.py --data_name delaney --train_strategy cd --prior_predict_type prior_similarity
```

**Step (2):** Visualize the feature similarity, the results will be saved in the folder `/figures/Prior prediction similarity`.

```bash
python 5-2-visualize_prior.py --data_name delaney --train_strategy cl --visualize_type prior_similarity
```

```bash
python 5-2-visualize_prior.py --data_name delaney --train_strategy cd --visualize_type prior_similarity
```

**Step (3):** Predict Feature distinctiveness using different priors.

```bash
python 5-1-prior_predict.py --data_name delaney --train_strategy cl --prior_predict_type prior_latent
```

```bash
python 5-1-prior_predict.py --data_name delaney --train_strategy cd --prior_predict_type prior_latent
```

**Step (4):** Visualize the feature distinctiveness, the results will be saved in the folder `/figures/Prior prediction distinctiveness`.

```bash
python 5-2-visualize_prior.py --data_name delaney --train_strategy cl --visualize_type prior_latent
```

```bash
python 5-2-visualize_prior.py --data_name delaney --train_strategy cd --visualize_type prior_latent
```

**Notes:** See more details in the section **5.4 Prior Prediction performance of contrastive prior** in the paper.

<details>
  <summary><b>Click here for the results!</b></summary>

<b>Feature similarity between samples.</b>

| ![Alts](figures/Prior%20prediction%20similarity/ESOL%20Uninformative%20Prior.JPG) | ![Alts](figures/Prior%20prediction%20similarity/ESOL%20Contrastive%20Prior.JPG) |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

<b>Feature distinctiveness between samples.</b>

| ![Alts](figures/Prior%20prediction%20distinctiveness/ESOL%20Uninformative%20Prior.JPG) | ![Alts](figures/Prior%20prediction%20distinctiveness/ESOL%20Contrastive%20Prior.JPG) |
|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|

</details>

<hr style="border: 0.1px">

## Enhance the task-specific posterior on customized dataset

**We recommend you to run the scripts above to reproduce our experiments before attempting to train the model using your custom dataset to familiarize yourself with the pre-training and posterior inference processes of the CPBayesMPP.**

**Step (1): Prepare your pre-training dataset.**

Please refer to `pubchem-1m-clean.csv` file <u>[here](https://drive.google.com/file/d/1FO_otxK3WHA629Xu1oseDWJPPJCUbgu5/view?usp=sharing)</u> to store your unlabeled dataset in rows according to the following format:

|        pretraining_dataset.csv         |
|:--------------------------------------:|
|                 smiles                 |
| CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1 |
|     Nc1ccc(NC(=O)CSc2cccc(F)c2)cc1     |
|   COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC   |
|      OCCn1cc(CNc2cccc3c2CCCC3)nn1      |
|    Nc1nc2ccc(F)c(F)c2n1Cc1ccccc1Br     |
|                  ...                   |

Then, run the script [dataset/prepare_pretrain_cache.py](dataset/prepare_pretrain_cache.py) to generate contrastive tasks cache for accelerating the pre-training process.

**Step (2): Pre-training for contrastive prior.**

Run the script [0-pretrain.py](0-pretrain.py) to learn pre-train the model and replace the parameter `--pretrain_data_name` with your own dataset name.

```bash
python 0-pretrain.py --batch_size 512 --epochs 50 --save_dir results/pretrain --pretrain_data_name your_pretrain_data_name --use_pretrain_data_cache
```

**Step (3): Prepare downstream datasets.**

Save your downstream datasets in folder `/dataset` by following format:

**regression_downstream_dataset.csv**

|         smiles         | label |
|:----------------------:|:-----:|
|    Clc1ccc2ccccc2c1    | -0.77 |
| Cc1occc1C(=O)Nc2ccccc2 | 5.14  |
|  CC(C)=CCCC(C)=CC(=O)  | -2.06 |
|          ...           |  ...  |

**classification_downstream_dataset.csv**

|             smiles              | label |
|:-------------------------------:|:-----:|
|  CCOc1ccc2nc(S(N)(=O)=O)sc2c1   |   0   |
|    CCN1C(=O)NC(c2ccccc2)C1=O    |   0   |
| CCCN(CC)C(CC)C(=O)Nc1c(C)cccc1C |   1   |
|               ...               |  ...  |

**Step (4): Infer task-specific posterior.**

Run the script [1-train.py](1-train.py) to infer the task-specific posterior and replace the parameter `--data_name` with your own dataset name.

```bash
python 1-train.py --data_name your_dataneme --train_strategy cl --epoch 200 --split_type random --split_sizes 0.5 0.2 0.3
```

**Step (5):  Posterior Prediction with
Uncertainty Evaluation.**

Run the script [3-1-uncertainty_predict.py](3-1-uncertainty_predict.py) to predict unseen molecules with uncertainty.

```bash
python 3-1-uncertainty_predict.py --data_name your_dataname --train_strategy cl --split_type random
```

The result format will be as follows:

|      smiles      | preds | aleatoric uncertainty | epistemic uncertainty |
|:----------------:|:-----:|:---------------------:|:---------------------:|
| COc2ncc1nccnc1n2 | -1.15 |         0.43          |         0.12          |
|   Brc1ccccc1Br   | -3.42 |         0.08          |         0.03          |
|    O=C1CCCCC1    | -0.53 |         0.10          |         0.06          |
|       ...        |  ...  |          ...          |          ...          |

## Thanks

Thanks for the support of the following repositories:

|                        Source                        |                                   Detail                                    |
|:----------------------------------------------------:|:---------------------------------------------------------------------------:|
|         https://github.com/chemprop/chemprop         |              Implementation of message passing neural network               |
| https://github.com/gscalia/chemprop/tree/uncertainty |                  Implementation of concrete dropout layer                   |
|          https://github.com/yuyangw/MolCLR           | Implementation of augmentation strategies for contrastive molecules samples |
|      https://github.com/google-research/simclr       |                       Implementation of NT-Xent Loss                        |

## Cite us

If you find this work useful to you, please cite our paper:

```
@article{XXX,
  title={Contrastive Prior Enhances the Performance of Bayesian Neural Network-based Molecular Property Prediction},
  author={XXX},
  journal={XXX},
  year={2024}
}
```
