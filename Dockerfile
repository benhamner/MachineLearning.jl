#
# MachineLearning.jl Dockerfile
#
# https://github.com/benhamner/MachineLearning.jl/tree/master/Dockerfile
#

# Pull base image.
FROM ubuntu:14.04

# Install Julia and clone MachineLearning.jl
RUN  cd /
RUN  apt-get install git software-properties-common -y
RUN  add-apt-repository ppa:staticfloat/julia-deps -y
RUN  add-apt-repository ppa:staticfloat/julianightlies -y
RUN  apt-get update -qq -y
RUN  apt-get install libpcre3-dev julia -y
RUN  julia -e 'Pkg.init()'
RUN  julia -e 'Pkg.clone("https://github.com/dcjones/Showoff.jl")'
RUN  julia -e 'Pkg.clone("https://github.com/benhamner/MachineLearning.jl"); Pkg.checkout("Gadfly"); Pkg.resolve()'
RUN  julia -e 'Pkg.pin("MachineLearning")'
RUN  julia -e 'using MachineLearning; @assert isdefined(:MachineLearning); @assert typeof(MachineLearning) === Module'
