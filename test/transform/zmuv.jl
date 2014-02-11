using Base.Test
using MachineLearning

x = [1.0 2.0]
zmuv = Zmuv([1.0 3.0],[1.0 1.0])
@test transform(zmuv, x)==[0.0 -1.0]
@test transform(zmuv, vec(x))==[0.0;-1.0]

zmuv = Zmuv([1.0 3.0],[5.0 2.0])
@test transform(zmuv, x)==[0.0 -0.5]

zmuv = Zmuv([1.0 3.0],[0.0 0.0])
@test transform(zmuv, x)==[0.0 0.0]

zmuv = fit([1.0 2.0], ZmuvOptions())
@test zmuv.means==[1.0 2.0]
@test zmuv.stds==[0.0 0.0]

zmuv = fit([1.0 2.0;3.0 2.0], ZmuvOptions())
@test zmuv.means==[2.0 2.0]
@test zmuv.stds==[sqrt(2.0) 0.0]