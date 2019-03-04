from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch

try:
    import mxnet
    has_mxnet = True
except ImportError:
    has_mxnet = False


def _convert_bn(k):
    aux = False
    if k == 'bias':
        add = 'beta'
    elif k == 'weight':
        add = 'gamma'
    elif k == 'running_mean':
        aux = True
        add = 'moving_mean'
    elif k == 'running_var':
        aux = True
        add = 'moving_var'
    else:
        assert False
    return aux, add


def convert_from_mxnet(model, checkpoint_prefix, debug=False):
    _, mxnet_weights, mxnet_aux = mxnet.model.load_checkpoint(checkpoint_prefix, 0)
    remapped_state = {}
    for state_key in model.state_dict().keys():
        k = state_key.split('.')
        aux = False
        mxnet_key = ''
        if k[0] == 'features':
            if k[1] == 'conv1_1':
                # input block
                mxnet_key += 'conv1_x_1__'
                if k[2] == 'bn':
                    mxnet_key += 'relu-sp__bn_'
                    aux, key_add = _convert_bn(k[3])
                    mxnet_key += key_add
                else:
                    assert k[3] == 'weight'
                    mxnet_key += 'conv_' + k[3]
            elif k[1] == 'conv5_bn_ac':
                # bn + ac at end of features block
                mxnet_key += 'conv5_x_x__relu-sp__bn_'
                assert k[2] == 'bn'
                aux, key_add = _convert_bn(k[3])
                mxnet_key += key_add
            else:
                # middle blocks
                if model.b and 'c1x1_c' in k[2]:
                    bc_block = True  # b-variant split c-block special treatment
                else:
                    bc_block = False
                ck = k[1].split('_')
                mxnet_key += ck[0] + '_x__' + ck[1] + '_'
                ck = k[2].split('_')
                mxnet_key += ck[0] + '-' + ck[1]
                if ck[1] == 'w' and len(ck) > 2:
                    mxnet_key += '(s/2)' if ck[2] == 's2' else '(s/1)'
                mxnet_key += '__'
                if k[3] == 'bn':
                    mxnet_key += 'bn_' if bc_block else 'bn__bn_'
                    aux, key_add = _convert_bn(k[4])
                    mxnet_key += key_add
                else:
                    ki = 3 if bc_block else 4
                    assert k[ki] == 'weight'
                    mxnet_key += 'conv_' + k[ki]
        elif k[0] == 'classifier':
            if 'fc6-1k_weight' in mxnet_weights:
                mxnet_key += 'fc6-1k_'
            else:
                mxnet_key += 'fc6_'
            mxnet_key += k[1]
        else:
            assert False, 'Unexpected token'

        if debug:
            print(mxnet_key, '=> ', state_key, end=' ')

        mxnet_array = mxnet_aux[mxnet_key] if aux else mxnet_weights[mxnet_key]
        torch_tensor = torch.from_numpy(mxnet_array.asnumpy())
        if k[0] == 'classifier' and k[1] == 'weight':
            torch_tensor = torch_tensor.view(torch_tensor.size() + (1, 1))
        remapped_state[state_key] = torch_tensor

        if debug:
            print(list(torch_tensor.size()), torch_tensor.mean(), torch_tensor.std())

    model.load_state_dict(remapped_state)

    return model

parser = argparse.ArgumentParser(description='MXNet to PyTorch DPN conversion')
parser.add_argument('checkpoint_path', metavar='DIR', help='path to mxnet checkpoints')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')


