#!/usr/bin/env python
import os
import json
import click
import boto3
from soocii_services_lib.click import CiTools, build_soocii_cli, bash

REPOSITORY = 'xavier'


class XavierCiTools(CiTools):
    def build_docker_image(self, version, label):
        with BackendSecretsInS3():
            return super().build_docker_image(version, label)


class BackendSecretsInS3:
    def __init__(self):
        self.bucket = 'soocii-secret-config-tokyo'
        self.secret_dir = './conf/'
        self.secret_files = json.load(open(os.path.join(self.secret_dir, 'secrets_conf.json'), 'r'))

    def __enter__(self):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)

        for key in self.secret_files:
            bucket.download_file(
                key,
                os.path.join(self.secret_dir, key)
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # keep secret files on local host for development. Otherwise, docker-compose up will mount ./app into docker
        # and secret files in docker will disappear.
        # for f_name in self.secret_files:
        #     os.remove(os.path.join(self.secret_dir, f_name))
        pass


@click.group(chain=True)
def development():
    pass


tools = XavierCiTools('xavier')
soocii_cli = build_soocii_cli(tools)

cli = click.CommandCollection(sources=[development, soocii_cli])

if __name__ == '__main__':
    cli()
