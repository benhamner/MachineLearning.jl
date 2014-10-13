using Base.Test
using MachineLearning

x = [("ben",),("ben",)]
hash_vect = HashVectorizer(HashVectorizerOptions(16))
m = transform(hash_vect, x)
@test m[1,:]==m[2,:]
@test sum(m[1,:])==1

x = [("ben",), ("ben","dog")]
m = transform(hash_vect, x)
@test sum(abs(m[2,:]-m[1,:]))==1

x = [("ben","cat","dog"), ("ben","dog","cat")]
m = transform(hash_vect, x)
@test m[1,:]==m[2,:]
@test sum(m)==6