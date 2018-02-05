FROM python:3.6

# Setup container version
ARG version
ENV CONTAINER_VERSION ${version}

RUN pip install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /var/www/xavier

COPY ./ /var/www/xavier
WORKDIR /var/www/xavier

RUN python -c 'import imageio; imageio.plugins.ffmpeg.download()'
