docker run -d -v /home/nandanrao:/starspace/mount --name sentencespace nandanrao/starspace Starspace/starspace train -trainFile mount/${1} -model mount/${2} -dim 100 -ngrams 1 -minCount 20 -trainMode 3 -fileFormat labelDoc -thread 64 -epoch 7 -saveEveryEpoch 1

