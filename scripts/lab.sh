docker run -d --user root -e NB_UID=5000 --name lab -v /home/nandanrao:/home/jovyan/work -p 8888:8888 jupyter/datascience-notebook start.sh jupyter lab
