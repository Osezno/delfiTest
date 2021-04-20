FROM gcr.io/google-appengine/python

# Create a virtualenv for dependencies. This isolates these packages from
# system-level packages.
# Use -p python3 or -p python3.7 to select python version. Default is version 2.

RUN virtualenv /env -p  3.7

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
RUN /env/bin/python -m pip install --upgrade pip
# RUN 
# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
#RUN apt-get update

# RUN apt-get update \
#         && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile-dev libsndfile1-dev -y \
#         && pip3 install pyaudio
# RUN apt-get update && apt-get install -y \
#         vim \
#         curl \
#         wget \
#         git \
#         make \
#         netcat \
#         python \
#         python2.7-dev \
#         g++ \
#         bzip2 \
#         binutils
###############################################################################
RUN apt-get install -y portaudio19-dev libopenblas-base libopenblas-dev pkg-config git-core cmake python-dev liblapack-dev libatlas-base-dev libblitz0-dev libboost-all-dev libhdf5-serial-dev libqt4-dev libsvm-dev libvlfeat-dev  python-nose python-setuptools python-imaging build-essential libmatio-dev python-sphinx python-matplotlib python-scipy
# additional dependencies
RUN apt-get install -y \
        libasound2 \
        libasound-dev \
        libssl-dev
RUN apt-get install -y alsa-base alsa-utils
RUN apt-get install -y pulseaudio
RUN pip install pyaudio
# RUN pip install --upgrade google-cloud-speech
#run --device /dev/snd:/dev/snd no se puede configurar desde aqui


# RUN apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 python-all-dev python3-all-dev python3-pyaudio  pulseaudio -y 
# RUN -it --rm \
# 		 --device /dev/snd \
# 		 -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
# 		 -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
# 		 -v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
# 		 -v /media/dyan/project/projects/voice/:/data/voice \
# 		--name python-speech-recognition python-speech-recognition-app
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ADD startup.sh /app/sartup.sh
# RUN /app/sartup.sh
ADD . /app


# Run a WSGI server to serve the application. gunicorn must be declared as
# a dependency in requirements.txt.
CMD gunicorn -b  :$PORT main:app
