## Fourier Neural Operator (FNO) on Lassen

This repo contains code to train/test FNO (U-NO) on [LLNL's Lassen](https://hpc.llnl.gov/hardware/compute-platforms/lassen) with distributed training. This is the main code used in the paper Multi-fidelity Fourier Neural Operator for Fast Modeling of Large-Scale Geological Carbon Storage in the Journal of Hydrology [arXiv version](https://arxiv.org/abs/2308.09113).

### [Running the code](#running)

1. prepare training data using 01_convert_data.ipynb  
2. run training using main.py  
3. run the test using run_test.py  
4. visualize the test results using visualize_test_results.ipynb  

Note: The code to generate the data has been released in the [Data Generation Repo](https://github.com/tang39/clastic_shelf_GEOS)

### [Folder structures](#structures)

The code is organized in the following way:

├── src    
│   ├── utils    
│   │   ├── 01_convert_data.ipynb    
│   │   ├── data_utils.py    
│   │   ├── distributed_utils.py    
│   │   ├── training_utils.py    
│   ├── main.py: the main training script     
│   ├── models: folder contains the models     
│   │   ├── model_utils.py    
│   │   ├── fno_3d.py: original FNO model   
│   │   ├── uno_3d.py: U-shaped FNO model   
│   ├── run_test.py: script to run the test after training   
│   ├── visualize_test_results.ipynb: notebook to visualize the test results    
├── scripts    
│   ├── distributed_run.sh: shell script to run on the cluster    
├── output: folder contains the output files     
├── README.md    
└── .gitignore  

### [License](#license)

This code is provided under the MIT License.

```text
 Copyright (c) 2023 Qingkai Kong/Hewei Tang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### [Acknowledgement](#acknowledgement)
```text

This work was completed as part of the Science-informed Machine learning to
Accelerate Real Time decision making for Carbon Storage (SMART-CS) Initiative
(edx.netl.doe.gov/SMART). Support for this initiative came from the U.S.
DOE Office of Fossil Energy’s Carbon Storage Research program. Part of the
implemented pipeline was supported by the LLNL-LDRD Program under
Project No. 23-FS-021.

For the original FNO model or the U-NO model please refer to the following
two repos:

# FNO: https://github.com/zongyi-li/fourier_neural_operator
# U-NO: https://github.com/ashiq24/UNO

We also worked on top of the code based on the version used in the following
two papers with the help from the authors.

Y. Yang, A. F. Gao, K. Azizzadenesheli, R. W. Clayton and Z. E. Ross,
"Rapid Seismic Waveform Modeling and Inversion With Neural Operators,"
in IEEE Transactions on Geoscience and Remote Sensing,
vol. 61, pp. 1-12, 2023, Art no. 5906712, doi: 10.1109/TGRS.2023.3264210.

Yan Yang, Angela F. Gao, Jorge C. Castellanos, Zachary E. Ross,
Kamyar Azizzadenesheli, Robert W. Clayton;
Seismic Wave Propagation and Inversion with Neural Operators.
The Seismic Record 2021;; 1 (3): 126–134. doi: https://doi.org/10.1785/0320210026

```

### [Disclaimer](#disclaimer)
```text
  This work was performed under the auspices of the U.S. Department of Energy
  by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
```

``Release Info: LLNL-CODE-858160``
