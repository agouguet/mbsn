
Install conda : https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

conda create --name mbsn_env python=3.10.12

conda activate mbsn_env

pip install -e .

python src/mbsn/main.py