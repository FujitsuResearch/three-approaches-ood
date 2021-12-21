docker build --build-arg HTTP_PROXY=$HTTP_PROXY \
       --build-arg http_proxy=$http_proxy \
       --build-arg HTTPS_PROXY=$HTTPS_PROXY \
       --build-arg https_proxy=$https_proxy \
       -t three_approaches ./
