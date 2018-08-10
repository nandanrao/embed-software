docker run --user root -e UID=5000 --rm -v /home/nandanrao:/home/jovyan/work jupyter/datascience-notebook jupyter nbconvert work/test-sentence-embeddings.ipynb --template work/report.tpl
