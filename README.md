# fin_gen_dl
This repo is the implementation of our paper called "FinGenDL: A Framework for Evolving Time-series Based Neural Networks through Augmenting Topologies".

### Installation instructions
These installation instructions assume that you have anaconda. and RabbitMQ. If this is not the case please download RabbitMQ via https://www.rabbitmq.com/news.html#2021-05-04T18:00:00+00:00. Please download anaconda via https://www.anaconda.com/products/individual.

1. After cloning the repo locally and cd'ing into the repo, we recommend setting up a conda environment.
```
conda create -n fin_gen_dl_env python=3.7
```

2. Activate environment.
```
conda activate fin_gen_dl_env
```

3. Cd into the directory such that you are on the same dir level as setup.py. Then enter the following to install the dependencies.
```
pip install -e .
```

4. Install tensorflow
```
pip install tensorflow
```

### Running the genetic and aggregation layers system
1. Run RabbitMQ from the terminal
```
brew services start rabbitmq
```

2. Open 5 terminals and start the 5 evolution processes.
```
python3 /path/to/fin_gen_dl/evo_node.py --hlayers <1 thru 5 per shell>
```

3. Run the aggregator routine.
```
python3 /path/to/fin_gen_dl/aggregator.py --acc_th -0.30
```

4. Trigger the workflow.
```
python3 /path/to/fin_gen_dl/main.py
```
