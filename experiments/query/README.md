# Query Experiment
## Goal
Check if similar query has trasferability. 

Patameter in this Experiment:
- D: dataset
- Q: query
- P: placement/machine
- C: query configuration (Knobs)

Evaluation:
- A: accuracy
- L: latency


## Experiments 1 (Trasferability on Dataset)
Experiment: Given 2 Dataset, same Query. Could we get similar Accuracy & Latency under same Knob & same Placement. 

- What is 2 Different dataset?
    1. Temporal Different: Same video, different segment. 
    2. Spatial Differnent: Different video. 
- What is similar Accuracy & Latency? 

### Experiment Preperation:
1. git clone https://github.com/nutonomy/nuscenes-devkit.git
2. Download dataset nuimages
    ```
    mkdir nuimages
    cd nuimages
    download tar.gz
    tar -
    ```
3. Create environment
    ```
    conda create --name nuscenes python=3.7
    conda activate nuscenes 
    
    sudo apt-get update && sudo apt-get install libgl1
    ```
4. Install dependency
    ```
    pip install -r requirement.txt
    ```

### How to reproduce experiment


## Experiments 2 (Transferability on Placement)
Experiment: Given 2 same placement, same Dataset, different but similar Query, could we get similar Accuracy & Latency. 
- What is similar query
    - Real time object detection that focus on differnt object


## Experiments 3 (Transferability on Placement)
Experiment: Given 2 different placement, same Dataset, same Query, could we get similar Accuracy & Latency under similar placement. 
