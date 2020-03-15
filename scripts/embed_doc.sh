#!/bin/sh

# Takes 3 positional arguments: 
# 1. the trained starspace model files (binary format)
# 2. the content file that you want to embed (text, line delimited)
# 3. the output file for the embeddings (outputs a text file)

docker run -d --name embed  -v $PWD:/starspace/mount nandanrao/starspace /bin/bash -c "Starspace/embed_doc mount/${1} < mount/${2} | tail -n +5 > mount/${3}"
