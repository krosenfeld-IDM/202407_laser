FROM rockylinux:9

ENV INPUT_ROOT=Assets

COPY requirements.txt /requirements.txt

RUN dnf upgrade -y

RUN dnf install -y python python-pip python-devel \
    vim git

RUN pip install pip --upgrade 
RUN pip install -r requirements.txt
