## Setup OpenCrispr
To get started with the program, follow the steps below:

### 1. Create a Conda environment
First, create a new environment with Python 3.12:

```
conda create -n opencrispr-env python=3.12 -y
conda activate opencrispr-env
```

### 2. Clone the repository
Download the project from GitHub:

```
git clone https://github.com/sfkelder/opencrispr-repro-main.git
```

### 3. Install dependencies
Navigate into the project folder and install the required packages:

```
cd opencrispr-repro-main
pip install .
```

> [!NOTE]
> If the installation fails, you can use the included requirements.txt file, which 
> contains the exact dependency versions used for this project.

## Setup Pipeline
This repository includes Python files for further data processing, located in the [`scripts`](./scripts) folder.  

Running these files requires some additional setup. It is recommended to use a separate environment to avoid conflicts with OpenCrispr and other projects.

> [!IMPORTANT]
> Example pipeline files are located in the [`example_pipelines`](./example_pipelines) 
> folder. Make sure to change the paths inside the files to your local corresponding 
> file paths.

### 1. Create a Conda environment
First, create a new environment with Python 3.12:

```
conda create -n pipeline-env python=3.12 -y
conda activate pipeline-env
```

### 2. Install dependencies

Conda packages:
```
conda install -c conda-forge -c bioconda foldseek mmseqs2 viennarna -y
```

Correct torch and cuda verions:
```
python -m pip install torch==2.10.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

Pip dependencies:
```
python -m pip install fair-esm seaborn accelerate
```

### 3. Download ESMFold
The pipeline requires ESMFold as a dependency. 

```
git clone https://github.com/sara-nl/ESMFold_Snellius
```

> [!WARNING]
> In the current setup, ESMFold may require a different CUDA version than the one 
> previously installed.
>
> On an HPC system, this can be resolved by loading the appropriate module, as 
> illustrated in [`train_pipeline.sh`](./example_pipelines/train_pipeline.sh). 
> Alternatively, a separate conda environment can be created specifically for ESMFold 
> to isolate its dependencies from other software.
>
> This approach ensures that version conflicts are avoided and that both the pipeline 
> and ESMFold can run reliably.


