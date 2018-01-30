FROM python:3.6

# Setup container version
ARG version
ENV CONTAINER_VERSION ${version}

RUN pip install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /var/www/secretagent

COPY ./ /var/www/secretagent
WORKDIR /var/www/secretagent
