MachineLearning.jl
==================

[![Build Status](https://travis-ci.org/benhamner/MachineLearning.jl.png)](https://travis-ci.org/benhamner/MachineLearning.jl)
[![Coverage Status](https://img.shields.io/coveralls/benhamner/MachineLearning.jl.svg)](https://coveralls.io/r/benhamner/MachineLearning.jl?branch=master)
[![Package Evaluator](http://iainnz.github.io/packages.julialang.org/badges/MachineLearning_release.svg)](http://iainnz.github.io/packages.julialang.org/?pkg=MachineLearning&ver=release)

The MachineLearning package represents the very beginnings of an attempt to consolidate common machine learning algorithms written in pure Julia and presenting a consistent API. Initially, the package will be targeted towards the machine learning practitioner, working with a dataset that fits in memory on a single machine. Longer term, I hope this will both target much larger datasets and be valuable for state of the art machine learning research as well.

API Introduction
===============

    model = [2.0,1.0,-1.0]
    x_train = randn(1_000, 3)
    y_train = int(map(x->x>0, x_train*model))
    net = fit(x_train, y_train, classification_net_options())
    sample = [1.0, 0.0, 0.0]
    println("Ground truth: ", int(dot(sample,model)>0))
    println("Prediction:   ", predict(net, sample))

Algorithms Implemented
======================

 - Basic Decision Tree for Classification
 - Basic Random Forest for Classification
 - Basic Neural Network
 - Bayesian Additive Regression Trees

Other Helpers
=============

 - Train/Test split
 - Cross validation
 - Experiments
