FROM python:3.7.6
WORKDIR /app
COPY requirements.txt .
RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install pip
RUN pip install wheel
RUN pip install numpy==1.19.5
RUN apt-get -y install gfortran
RUN apt-get -y install libhdf5-dev libc-ares-dev libeigen3-dev
RUN apt-get -y install libatlas-base-dev libopenblas-dev libblas-dev
RUN apt-get -y install liblapack-dev
RUN pip install --upgrade setuptools
RUN pip install pybind11
RUN pip install Cython
RUN pip install h5py==3.1.0
#RUN wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl -q --show-progress
COPY . .
RUN pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl
RUN pip install -r requirements.txt

CMD ["python","-u","server.py"]
