FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
# for gnu screen
ENV SHELL /bin/bash

# Remove #s for the two lines below after setting your time zone, so that you can avoid an interruption in the process of building the environment.
#ENV TZ={your time zone here}
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \ 
    vim \
    xvfb \
    python3 \
    python3-pip \
    libopencv-dev \
    screen \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tqdm
RUN pip install opencv-python
RUN pip install pettingzoo==1.22
RUN pip install multi-agent-ale-py
RUN pip install AutoROM
RUN AutoROM -y
RUN pip install tensorboardX
RUN pip install tensorflow tensorflow_addons
RUN pip install scipy
RUN pip install scikit-image==0.17.2
RUN pip install ipython==7.19.0
RUN pip install moviepy
RUN pip install --upgrade pillow==9.5.0

RUN \
   echo 'alias python="/usr/bin/python3"' >> /root/.bashrc && \
   echo 'alias pip="/usr/bin/pip3"' >> /root/.bashrc && \
   source ~/.bashrc

WORKDIR /work
