# CELL
This is the repository for the resources of paper "CELL : Cluster based Ensembel Learning for Large Language Model"

## Abstract
In recent years, language models have made significant advancements following the introduction of Pretrained Language Models(PLMs) such as BERT, demonstrating remarkable performance. Research efforts various domains, encompassing structural changes, dataset variations, and training methodologies, have led to the development of numerous models exhibiting strong performance in Natural Language Processing(NLP) tasks.
Most recent models learn linguistic features and contextualized sentence presentation by pre-training through large datasets and fine-tuning to downstream tasks. Ensemble methods are also often employed to further enhance performance in these downstream tasks. However, existing ensemble methods are not out of the frameworks of Bagging and Boosting, and are inefficient.
In this paper, we introduce a novel approach called Cluster-based Ensemble Learning for Large Language Models(CELL) to enhance the performance of classification tasks efficiently, surpassing conventional ensemble methods in the context of Large Language Models(LLMs). We demonstrate how our model utilizes clusters to increase data diversity, improve the efficiency of learning by dividing the data, and ultimately enhance performance.


## Environment
```
Python 3.10.12
transformers==4.27.4
torch==2.0.0
```

## Dataset
```
python3 make_data.py \
    --task_name mrpc
```

## How can use?
```
python3 ./run_cell.py \
    --input_dir ./dataset/mrpc_train \
    --validation_dir ./dataset/mrpc_validation \
    --output_dir ./ \
    --final_model ./GLUE_base_model/bert_mrpc \
    --epochs 5 \
    --batch_size 16 \
    --use_gpu \
    --model_num 2\
    --cluster GaussianMixture..tied.random-from-data\
    --task_name GaussianMixture..tied.random-from-data_mrpc
```
