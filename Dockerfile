#
# MachineLearning.jl Dockerfile
#
# https://github.com/benhamner/MachineLearning.jl/tree/master/Dockerfile
#
# docker build -t="benhamner/machine_learning" .
# docker run -t -i benhamner/machine_learning /bin/bash

# Pull base image.
FROM ubuntu:14.04

# Install Julia and clone MachineLearning.jl
RUN  cd /
RUN  apt-get install git software-properties-common curl wget gettext libcairo2 libpango1.0-0 -y
RUN  add-apt-repository ppa:staticfloat/julia-deps -y
RUN  add-apt-repository ppa:staticfloat/julianightlies -y
RUN  apt-get update -qq -y
RUN  apt-get install libpcre3-dev julia -y
RUN  julia -e 'Pkg.init()'
RUN  julia -e 'Pkg.clone("https://github.com/dcjones/Showoff.jl"); Pkg.clone("https://github.com/benhamner/MachineLearning.jl"); Pkg.checkout("Gadfly"); Pkg.checkout("MachineLearning"); Pkg.pin("MachineLearning"); Pkg.resolve();'
RUN  julia -e 'using MachineLearning; @assert isdefined(:MachineLearning); @assert typeof(MachineLearning) === Module'

CMD ["julia", "/root/.julia/v0.4/MachineLearning/test/runtests.jl"]
