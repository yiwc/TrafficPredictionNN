
# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

## Updating Log

### Variables

sensor_ids, len=207, cont_sample="773869", a random 6-digit number\
adj_mx, shape=207,207 , if Identity, it is a eye(207)\
scaler, a variable maybe used in the later part to scale paras. It includes mean and std of the data


sensor_id_to_ind, 

adjinit， used in gwnet as addaptadj
```

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
```

iterator, for iter, (x, y) in iteretor:
```angular2
Iterator
```
### Per Iterate:
every iterate collect a data batch len of 64
trainx, shape=64,2,207,12\
trainy, shape=same as trainx\
```
trainx = torch.Tensor(x).to(device)
trainx= trainx.transpose(1, 3)
trainy = torch.Tensor(y).to(device)
trainy = trainy.transpose(1, 3)
metrics = engine.train(trainx, trainy[:,0,:,:])
train_loss.append(metrics[0])
train_mape.append(metrics[1])
train_rmse.append(metrics[2])
```

### Data Introduction
for example METR-LA\
x_train, shape=23974,12,207,2\
y_train, shape=23974,12,207,2\

- 207 nodes\
- 2 , eg. [signal_value     time_stamp] (by hour ? or minute? second?)\
in y, 2, eg. [speed time_stamp]\
- 12, Last Row is the latest .  12， similar to the memory. Once collect 12 data per time. 一次采集了12个时间序列,207个节点\
- 23974, time series max steps



x_val, shape=3425,12,207,2\
y_val, shape=3425,12,207,2\
x_test, shape= 6850,12,207,2\
y_test, shape= 6850,12,207,2





# ↓Following Content Comes From https://github.com/nnzhan/Graph-WaveNet
This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).

<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>



## Requirements
- python 3
- pytorch
- scipy
- numpy
- pandas
- pyaml


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Follow [DCRNN](https://github.com/liyaguang/DCRNN)'s scripts to preprocess data.

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Experiments
Train models configured in Table 3 of the paper.

```
ep=100
dv=cuda:0
mkdir experiment
mkdir experiment/metr

#identity
expid=1
python train.py --device $dv --gcn_bool --adjtype identity  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-only
expid=2
python train.py --device $dv --gcn_bool --adjtype transition --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#adaptive-only
expid=3
python train.py --device $dv --gcn_bool --adjtype transition --aptonly  --addaptadj --randomadj --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-backward
expid=4
python train.py --device $dv --gcn_bool --adjtype doubletransition  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-backward-adaptive
expid=5
python train.py --device $dv --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

```


