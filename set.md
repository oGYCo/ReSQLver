# ReSQLver
Learning SQL Reward Models via Progressive Edit-Based Comparisons


数据集：
```bash
sudo apt update
sudo apt install -y tmux git wget
cd data
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd ..
bash download_model.sh
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r requirements.txt
python add_question_id.py
bash run_full_pipeline.sh
```

```bash
python -m tree.build_tree_dataset \
    --model_path /home/taocy/yuchen/Qwen2.5-Coder-1.5B-Instruct \
    --input_file "/home/taocy/yuchen/data/dev_bird.json" \
    --dev_file "/home/taocy/yuchen/data/dev_20240627/dev.json" \
    --db_path "/home/taocy/yuchen/data/dev_20240627/dev_databases" \
    --output_dir "/home/taocy/yuchen/ReSQLver/tree_dataset" \
    --max_depth 5 \
    --sample_num 3 \
    --temperature 0.8 \
    --early_stop
```

测试：
```bash
python -m tree.build_tree_dataset \
    --model_path /home/taocy/yuchen/Qwen2.5-Coder-3B-Instruct \
    --input_file "/home/taocy/yuchen/data/dev_bird.json" \
    --dev_file "/home/taocy/yuchen/data/dev_20240627/dev.json" \
    --db_path "/home/taocy/yuchen/data/dev_20240627/dev_databases" \
    --output_dir "/home/taocy/yuchen/ReSQLver/outputs/tree_test" \
    --max_depth 5 \
    --sample_num 2 \
    --start_idx 0 \
    --end_idx 10 \
    --tensor_parallel_size 1 \
    --visible_devices 0,1,2,3
```
```bash
python evaluate.py \
    --base_model Qwen2.5-Coder-1.5B-Instruct \
    --adapters qwen-reward-model-sql-train \
    --test_data final_data_test.json \
    --batch_size 64
```
```bash
python -m eval \
    --tree_file /home/taocy/yuchen/ReSQLver/outputs/tree_dataset/part3/20251129_094036/all_trees.json \
    --db_root /home/taocy/yuchen/data/dev_20240627/dev_databases \
    --output verification_report.json
```
python -m eval \
    --tree_file output/tree_12.json \
    --db_root data/train/train_databases \
    --output verification_report.json
# 树的数据结构
每棵树针对一个question，根节点是的node_id是node_1_1

整棵树的数据结构如下：
```json
{
    //=====================当前树的公共元数据==================================
    "question_id": "0",//整棵树所处理的问题的id，与dev.json中的question_id对应
    "db_id": "california_schools",//整棵树所处理的问题的db_id，与dev.json中的db_id对应
    "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",//BIRD数据集的当前question的evidence信息，用于辅助模型生成sql代码，与dev.json中的evidence对应
    "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",//整棵树所处理的问题，与dev.json中的question对应
    "difficulty": "simple",//整棵树所处理的问题的难度,与dev.json中的diffculty对应
    "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",//当前问题的标准答案，用来验证每个节点由模型生成的generated_sql是否正确，与dev.json中的SQL对应

    //==========================树的节点信息==================================
    //扁平化的节点列表 (用字典存储，Key是node_id)
    //node_id: node_d_j:d表示该节点的深度，j表示这个节点在这个深度的第几个位置
    //这个node_initial是个特殊节点，用来生成根节点的sql代码
    "node_initial": {
        "input_seq": "..."//用于生成初始根节点sql代码的提示词，input_seq是初始代码生成提示词模板（即会给定任务概述、数据库引擎、数据库的schema信息、evidence、question、对要生成sql的说明、输出格式、一步一步思考），其中主要的信息来自dev_bird.json的"input_seq"段。
    },
    "nodes": {
        "node_1_1": {//根节点
            "parent_id": "node_initial",
            //...
        },
        "node_2_1": { //node_id为node_2_1的节点
            "parent_id": "node_1_1",//当前节点的父节点
            "d": 2,//当前节点的深度,
            "j": 1,//当前节点在这个深度的第几个位置
            "children_num": 3,//当前节点的子节点个数，通常理解当前节点在进行下一次采样时的采样次数，也就是基于当前节点采样一次就会生成一个子节点
            "children_ids": ["node_3_1", "node_3_2", "node_3_3"],//当前节点的子节点
            "generated_sql": "...",//针对这个任务由模型采样生成的sql代码，这个sql代码是由当前节点的父节点的input_seq进行生成的
            "execution_feedback": "...",//执行当前节点生成的generated_sql代码获得的反馈信息，包括但不限于没有执行成功，执行成功但是执行之后的结果与ground_truth的sql代码的执行结果不一致等等
            "is_correct": false,//当前节点的generated_sql代码的执行结果是否正确，如果这个节点是正确的就不需要再根据这个节点继续往后面采样生成了，如果错误且d小于最大深度那么后续就需要根据这个节点的信息继续采样生成其子节点
            "input_seq":"...",//基于当前节点要继续往下面采样时给模型的输入。如果当前节点的is_correct == true，则当前节点的input_seq为空，说明不需要往下继续采样了。如果当前节点的is_correct == false，则当前节点的的input_seq是选择渐进编辑提示词模板（即会给定任务描述，数据库引擎，数据库的schema信息，evidence、question、当前节点的generated_sql代码、当前节点的generated_sql代码的执行反馈也就是excution_feedback的信息，让模型对当前的错误代码进生成相应的修订后的sql代码，输出格式，一步一步思考），其中还是会有来自dev_bird,json的"input_seq"段的信息
            "revision_distance": 3//修订距离，结果由树的叶节点往上进行“反向传播”得到。如果当前节点的is_correct == true，那么这个修订距离就是0，代表不再需要继续往后采样修订sql了，如果是叶子节点且is_correct == false，这个修订距离是inf，这里用999来代表inf，如果不是叶子节点且is_correct == false，则当前节点的修订距离等于其所有子节点中修订距离的最小值+1，如果所有子节点的修订距离都是inf，则当前节点的修订距离仍然是inf。特别的如果整棵树的根节点的修订距离是inf，我们认为这颗树是一颗bad tree，也就是其中的节点没有能够正确执行的sql
        },
        //...更多节点
    },
    //========================树的全局属性=============================
    "max_depth": 5,//树的最大深度
    "tree_status": "good"//good: 包含至少一个正确节点; bad:全是错误节点 (对应 root distance = 999)
}
```

步骤：
- 先用父节点的input_seq来采样一次，生成三个子节点，子节点有独立的generated_sql
- 然后执行每个子节点的sql，获得反馈（有可能没执行成功，也可能执行成功后跟标准答案的执行结果不一样）和is_correct
- 构造子节点的input_seq(给定系统提示词、数据引擎、任务、db信息，要求(要修改，根据它当前的错误的generated_sql、反馈信息))来生成子节点的修订后的sql代码
- 反向传播计算revision_distance以及判断tree_status