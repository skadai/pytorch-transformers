# -*- coding: utf-8 -*-

# @File    : prepare_corpus.py
# @Date    : 2019-09-02
# @Author  : skym


import click
import json


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj.update(json.load(open('SUBTYPE.json')))


@cli.command()
@click.option('-i', '--paper-id', help='specify paper id', required=True)
@click.argument('output-file', type='str')
@click.pass_context
def gen_polar_corpus(ctx):
    print(ctx.obj['skincare'])


if __name__ == '__main__':
    cli(obj={})
