# GPS2Graph
Transform GPS trajectory data into graph data!

Latest version: 0.0.2

<img decoding="async" src="https://github.com/zachysun/Gps2graph/blob/main/imgs/display.ppng" width="800" height="364">

***

#### Notes:

1.raw data link for test [scrg](https://cse.hkust.edu.hk/scrg/)

2.inputs: raw trajectory data; outputs: adjacency matrix, feature matrix, io matrix, distance matrix

3.io matrix: traffic flow and average travel time from one node to another over a period of time(default is 3 hours, you can set it yourself)

4.provide speed calculation function

***

#### Usage:

1.create and activate your conda environment

```python
conda create -n name python==3.8
activate name
```

2.download required packages

```python
pip install -r requirements.txt
```

3.create a folder named 'saved_files' to save the results

4.running python file

```python
python gps2graph.py
```

5.upload files and set parameters

***

#### TODO:

- [x] 1.add  transformation method with clustering
- [x] 2.add more settable parameters
- [ ] 3.segment feature matrix to train set, validation set, and test set

