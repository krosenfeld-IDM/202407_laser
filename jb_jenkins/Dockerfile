FROM rockylinux:9

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
RUN git clone -b jb_modulify --single-branch https://github.com/InstituteforDiseaseModeling/laser.git && cd laser/jb && pip install -e .
RUN cd laser/jb/src/idmlaser && make update_ages.so
