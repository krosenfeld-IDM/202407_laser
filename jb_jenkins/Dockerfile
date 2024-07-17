FROM rockylinux:9

ENV INPUT_ROOT=Assets

COPY requirements.txt /requirements.txt

RUN dnf upgrade -y

RUN dnf install -y \
    python \
    python-pip \
    python-devel \
    gcc-c++ \
    epel-release \
    vim \
    git \
    sudo

RUN pip install pip --upgrade 
RUN pip install -r requirements.txt
RUN git clone https://github.com/krosenfeld-IDM/202407_laser.git && cd 202407_laser/jb_src && pip install -e .
RUN cd 202407_laser/jb_src/src/idmlaser && make update_ages.so
