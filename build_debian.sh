sudo apt install -y libavcodec-dev libeigen3-dev libavformat-dev libavutil-dev libavresample-dev libsamplerate0-dev libyaml-dev libtaglib-cil* libtag1-dev python3-taglib libfftw3-* libchromaprint-*
# https://github.com/supermihi/pytaglib
./waf configure --with-example=streaming_extractor_music
./waf
pip install pytaglib
