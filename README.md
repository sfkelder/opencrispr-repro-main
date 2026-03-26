## Setup
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

> **Note:** 
> If the installation fails, you can use the included requirements.txt file, which 
> contains the exact dependency versions used for this project.