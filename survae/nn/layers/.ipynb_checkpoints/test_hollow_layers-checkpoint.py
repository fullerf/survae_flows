import torch
import torch.nn as nn
import einops
import math
from hollow_layers import (HollowConv2d, HollowConv1d, 
                          HollowLinear, SplitHollowConv1d,
                          SplitHollowConv2d, SplitHollowLinear)


# A Hollow Layer is one where x is mapped to y, such that the dimensionality of y matches x
# or is a positive integer multiple of x. Furthermore, to be "hollow" means that for some
# positive integer `k`, we can write `y[k*i] = f(x[j for j not = i])`. Let `J = jacobian(y,x)`. Then
# `J[k*i,i]` for all `k` must be exactly zero.
#
# That jacobian condition is what we test here

def test_hollow_conv2d():
    c,h,w = 3,8,8
    ks = [2,3]
    
    fake_img = torch.randn(1,c,h,w)
    kernel_sizes = [2,3,4,5]
    tests = []
    for kernel_size in kernel_sizes:
        for k in ks:
            out_channels = c*k
            f = HollowConv2d(c,out_channels,kernel_size,padding='same')
            J = torch.autograd.functional.jacobian(f, fake_img)
            J = einops.rearrange(J, 'b c h w a i j k -> b a (c h w) (i j k)')
            J = J.squeeze(0).squeeze(0) # now just a matrix, more like we'd expect
            for split in J.split(c*h*w):
                d = torch.diag(split)
                tests.append(torch.allclose(d, torch.zeros_like(d)))
    assert all(tests)
    


test_hollow_conv2d()


def test_hollow_conv1d():
    c,w = 3,8
    ks = [2,3]
    
    fake_spec = torch.randn(1,c,w)
    kernel_sizes = [2,3,4,5]
    tests = []
    for kernel_size in kernel_sizes:
        for k in ks:
            out_channels = c*k
            f = HollowConv1d(c,out_channels,kernel_size,padding='same')
            J = torch.autograd.functional.jacobian(f, fake_spec)
            J = einops.rearrange(J, 'b c w a i k -> b a (c w) (i k)')
            J = J.squeeze(0).squeeze(0) # now just a matrix, more like we'd expect
            for split in J.split(c*w):
                d = torch.diag(split)
                tests.append(torch.allclose(d, torch.zeros_like(d)))
    assert all(tests)
    


test_hollow_conv1d()


def test_hollow_linear():
    w = 8
    ks = [2,3]
    fake_spec = torch.randn(1,w)
    tests = []
    for k in ks:
        out_dim = k*w
        f = HollowLinear(w,out_dim)
        J = torch.autograd.functional.jacobian(f, fake_spec)
        J = einops.rearrange(J, 'b o a i -> b a o i')
        J = J.squeeze(0).squeeze(0) # now just a matrix, more like we'd expect
        for split in J.split(w):
            d = torch.diag(split)
            tests.append(torch.allclose(d, torch.zeros_like(d)))
    assert all(tests)
    


test_hollow_linear()


# We want to test the split layers; these are simple. They just wrap the torch.split function, essentially defaulting some parameters.

def test_split_hollow_conv2d():
    c,h,w = 3,8,8
    ks = [2,3]
    
    fake_img = torch.randn(1,c,h,w)
    kernel_sizes = [2,3,4,5]
    tests = []
    for kernel_size in kernel_sizes:
        for k in ks:
            out_channels = c*k
            f = HollowConv2d(c,out_channels,kernel_size,padding='same')
            splits = SplitHollowConv2d(k)(f(fake_img))
            for split in splits:
                tests.extend([s == t for s,t in zip(split.shape,[1,c,h,w])])
    assert all(tests)


def test_split_hollow_conv1d():
    c,w = 3,8
    ks = [2,3]
    
    fake_spec = torch.randn(1,c,w)
    kernel_sizes = [2,3,4,5]
    tests = []
    for kernel_size in kernel_sizes:
        for k in ks:
            out_channels = c*k
            f = HollowConv1d(c,out_channels,kernel_size,padding='same')
            splits = SplitHollowConv1d(k)(f(fake_spec))
            for split in splits:
                tests.extend([s == t for s,t in zip(split.shape,[1,c,w])])
    assert all(tests)


def test_split_hollow_linear():
    w = 8
    ks = [2,3]
    fake_spec = torch.randn(1,w)
    tests = []
    for k in ks:
        out_dim = k*w
        f = HollowLinear(w,out_dim)
        splits = SplitHollowLinear(k)(f(fake_spec))
        for split in splits:
                tests.extend([s == t for s,t in zip(split.shape,[1,w])])
    assert all(tests)


test_split_hollow_linear()


