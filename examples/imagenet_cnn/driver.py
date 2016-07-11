import numpy as np
import ray
import ray.services as services
import os
import tensorflow as tf
import ray.datasets.imagenet as imagenet
import argparse
import boto3
import random

import functions

def one_hot(x):
  zero = np.zeros([1000])
  zero[x] = 1.0
  return zero

num_workers = 3
batchnum = 0
weights = []

parser = argparse.ArgumentParser(description="Parse information for data loading.")
parser.add_argument("--s3-bucket", default="sparknet", type=str, help="Name of the bucket that contains the image data.")
parser.add_argument("--key-prefix", default="ILSVRC2012_img_train/n015", type=str, help="Prefix for files to fetch.")
parser.add_argument("--label-file", default="train.txt", type=str, help="File containing labels")

args = parser.parse_args()
worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "worker.py")
services.start_ray_local(num_workers=num_workers, worker_path=worker_path)
# Y = np.asarray(map(one_hot,np.random.randint(0, 1000, size=[1000]))) # Random labels and images in lieu of actual imagenet data
# X = np.random.uniform(size=[1000, 224, 224, 3])
s3 = boto3.resource("s3")
imagenet_bucket = s3.Bucket(args.s3_bucket)
objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
images = [obj.key for obj in objects.all()]

imagenet = ray.get(imagenet.load_tarfiles_from_s3(args.s3_bucket, map(str, images), [256, 256]))
print imagenet[0]
X = map(lambda img: img[0], imagenet)

s5 = boto3.client("s3")
labels = s5.get_object(Bucket=args.s3_bucket, Key=args.label_file)
lines = labels["Body"].read().split("\n")
imagepairs = map(lambda line: line.split(" ", 2), lines)
imagenames = map(lambda img: ray.get(img[1]), imagenet)
Y = map(lambda imglist: ray.put(map(lambda imgname:int(filter(lambda x:imgname in x, imagepairs)[0][1]), imglist)), imagenames)
batches = zip(X,Y)
for placeholder in functions.placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
print "weights inited"
functions.sess.run(functions.assignment, feed_dict=dict(zip(functions.placeholders, weights)))
print "Weights passed"
while True:
  results = []
  print "Start of loop"
  weights = functions.sess.run(functions.parameters)
  print "Weights recieved"
  weightrefs = map(ray.put, weights)
  print "Weightrefs created"
  for i in range(num_workers):
    functions.update_weights(*weightrefs)
  print "Workers updated"
  for i in range(num_workers):
    curbatch = random.choice(batches)
    xref = curbatch[0]
    yref = curbatch[1]
    results.append(functions.compute_grad(xref, yref))
  print "Grads recieved"
  actualresult = map(lambda x: map(ray.get, x), results)
  grads = [np.asarray([gradset[i] for gradset in actualresult]) for i in range(16)] # 16 gradients, one for each variable
  gradientvalues = map(lambda x: np.mean(x, axis=0), grads) # Taking mean over all the samples
  functions.sess.run(functions.application, feed_dict=dict(zip(functions.placeholders, gradientvalues))) # Feeding the new values in
  if (batchnum % 10 == 0):
    functions.print_accuracy(X, Y)
  print("End of batch {}".format(batchnum))
  batchnum += 1

services.cleanup()
