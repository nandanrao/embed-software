docker run --rm --name embed  -v $PWD:/starspace/mount nandanrao/starspace /bin/bash -c "Starspace/embed_doc mount/ss/sentencespace < mount/${1} | tail -n +5 > mount/${2}"
