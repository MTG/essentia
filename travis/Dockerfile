# This dockerfile defines the expected runtime environment before the project is installed
FROM ubuntu:latest

ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"
ENV DEBIAN_FRONTEND noninteractive
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:"/usr/local/config"

# Be sure to install any runtime dependencies
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
                       language-pack-en \
                       ca-certificates \
#                       qt5-default qtbase5-dev qtchooser qt5-qmake \  
                       build-essential apt-utils \
                       autotools-dev automake autoconf libtool-bin cmake \
                       python3-dev python3.6-dev python3-setuptools python3-pip \
                       python-dev python2.7-dev python-setuptools python-pip \
                       wget \
                       git swig doxygen \
                       libfftw3-dev libass-dev libfreetype6-dev \
  		       libsdl1.2-dev libtheora-dev libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
  		       libxcb-xfixes0-dev texinfo zlib1g-dev yasm libfdk-aac-dev libmp3lame-dev libopus-dev \
                       libsamplerate0-dev libtag1-dev libavcodec-dev libavformat-dev libavresample-dev \
                       libchromaprint-dev libyaml-dev \
                       pkg-config \
                   && rm -rf /var/lib/apt/lists/*

RUN pip3 install auditwheel
RUN pip install auditwheel

#RUN mkdir -p /tmp/yaml \
#    && cd /tmp/yaml \
#    && git clone https://github.com/yaml/libyaml.git \
#    && cd libyaml \
#    && ./bootstrap \
#    && ./configure && make && make install \
#    && cd /
                     
# Waf
# See https://github.com/waf-project/waf.git
RUN git clone https://github.com/waf-project/waf.git /tmp/waf \
    && cd /tmp/waf \
    && python3 ./waf-light configure --tools=swig,qt5,qt4 build \
    && cd /

# Gaia

#RUN qtchooser=qt5 
#ENV QT_SELECT=qt5

#RUN git clone https://github.com/sildeag/gaia /tmp/gaia    \
#    && cd /tmp/gaia \
#    && cp /tmp/waf/waf . \
#    && python3 ./waf configure --with-python-bindings --with-gaia-qt5 \ 
#    && cat /tmp/gaia/build/config.log \
#    && python3 ./waf \
#    && python3 ./waf install \
#    && cd / && rm -r /tmp/gaia

# Essentia
# See http://essentia.upf.edu/documentation/installing.html

RUN git clone https://github.com/sildeag/essentia /tmp/essentia \
    && cd /tmp/essentia \
    && cp /tmp/waf/waf . \
    && python waf configure --build-static --static-dependencies \
    && python waf \
    && python waf install \
#    && cp /tmp/essentia/build/src/examples/essentia_streaming_extractor_music_svm /code/hl_extractor/streaming_extractor_music_svm \
    && cd / \
#    && rm -r /tmp/essentia

    && cd /tmp/essentia/travis \
    && ls -l *.sh \
    && ./build_wheels_test.sh /tmp/essentia /usr/bin /usr/bin/python* /usr/local/bin /usr/local/bin/pip*
