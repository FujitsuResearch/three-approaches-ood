#FROM tensorflow/tensorflow:2.5.0
FROM tensorflow/tensorflow:2.5.0-gpu

COPY requirements.txt /tmp

RUN pip install --upgrade pip setuptools && \
    pip --no-cache-dir install -r /tmp/requirements.txt

RUN apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*