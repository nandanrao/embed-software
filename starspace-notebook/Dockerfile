#
# Starspace + Notebook Dockerfile
#

FROM jupyter/datascience-notebook

USER root

WORKDIR /

# install
RUN \
	apt-get update && apt-get install -y \
	autoconf \
    automake \
	libtool \
	build-essential \
	unzip \
	git \
	wget

# build boost lib
RUN \
 wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip && \
 unzip boost_1_63_0.zip && \
 mv boost_1_63_0 /usr/local/bin

# build starspace
RUN \
	git clone https://github.com/nandanrao/Starspace.git && \
	cd Starspace && \
	make && \
	make embed_doc && \
	make query_nn && \
	make print_ngrams && \
	make query_predict && \
    mv -t /usr/local/bin starspace embed_doc query_nn print_ngrams query_predict
