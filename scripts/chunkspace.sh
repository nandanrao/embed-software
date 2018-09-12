docker run -d -v /home/nandanrao:/starspace/mount --name docspace nandanrao/starspace Starspace/starspace train -trainFile mount/${1} -model mount/${2} -dim 100 -ngrams 1 -minCount 20 -trainMode 6 -negSearchLimit 100 -fileFormat labelDoc -thread 32 -initModel mount/${3}
