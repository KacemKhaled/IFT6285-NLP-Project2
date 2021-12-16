# IFT6285-NLP-Project2
Repository for the Project 2 for the NLP Course IFT6285 at the University of Montreal.

# using WSL :
```bash
cd /mnt/c/Users/kacem/Workspace/IFT6285/github-nlp/IFT6285-NLP-Project2
```

```bash
python -m spacy download en_core_web_sm
```
Preprocess training data:
```bash
python pre-process.py --min=5 --max=25
                    --no='" - -- # www http'
                    --nb=1000 --out=<fichier>.ref
                    --lower >! <fichier>.test
# Train kenlm model bigram on 99 files and build the binary files:
python reorder.py -f full -b 99 -m models/bigram -o 2
# Train kenlm model trigram on 99 files and build the binary files:
python reorder.py -f full -b 99 -m models/trigram -o 3
# Train kenlm model trigram on 99 files and build the binary files head=3 backoff=1 scoring=score
python reorder.py -f original -n 99 -m models/trigram_op -o 3  -s score -b 1  -d 3
# Show results example:
python show_results.py -n 30 -c -f news trigram

```

```bash
#install kenlm
pip install https://github.com/kpu/kenlm/archive/master.zip


```

Environment can be set with `conda` using:

```bash
conda env create -f environment.yml
# To activate this environment, use
conda activate nlp-project-env-conda
# To deactivate an active environment, use
conda deactivate
```
```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

```

To install requirements from requirements.txt

```bash
pip install -r requirements.txt
```

Updating an environment

```bash
# update the contents of your environment.yml file accordingly and then run the following command:
conda env update --prefix ./env --file environment.yml  --prune
# The --prune option causes conda to remove any dependencies that are no longer required from the environment.
```


To remove the environment:
```bash
 conda env remove --name nlp-project-env-conda
 ```