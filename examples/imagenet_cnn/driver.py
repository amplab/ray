import numpy as np
import ray
import ray.services as services
import os
import tensorflow as tf
import imagenet
import argparse
import boto3
import random

import functions


numofshuffles = 5
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

s3 = boto3.resource("s3")
imagenet_bucket = s3.Bucket(args.s3_bucket)
objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
images = [obj.key for obj in objects.all()]

imagenet = ray.get(imagenet.load_tarfiles_from_s3(args.s3_bucket, map(str, images), [256, 256]))
meanref = functions.compute_mean_image(imagenet)
X = map(lambda img: img[0], imagenet)
s3 = boto3.client("s3")
labels = s3.get_object(Bucket=args.s3_bucket, Key=args.label_file)
lines = labels["Body"].read().split("\n")
imagepairs = map(lambda line: line.split(" ", 2), lines)
imagepairs = ray.put(dict(map(lambda tup: (os.path.basename(tup[0]), tup[-1]), imagepairs)))
imagenames = map(lambda img: img[1], imagenet)
Y = map(lambda x: functions.convert(x, imagepairs), imagenames)
batches = zip(X,Y)
for i in range(numofshuffles):
  permuted_batches = np.random.permutation(batches)
  batches = map(lambda tup:ray.get(functions.shufflestuples(tup)), zip(permuted_batches,batches))
  print ray.get(batches[0][1])
print("Imagenet downloaded")
for placeholder in functions.placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
functions.sess.run(functions.assignment, feed_dict=dict(zip(functions.placeholders, weights)))
print("Weights passed")
while True:
  results = []
  print("Start of loop")
  weights = functions.sess.run(functions.parameters)
  print("Weights recieved")
  weightrefs = map(ray.put, weights)
  print("Weightrefs created")
  for i in range(num_workers):
    functions.update_weights(*weightrefs)
  if (batchnum % 100 == 0):
    temp = random.choice(batches)
    xref = temp[0]
    yref = temp[1]
    print ray.get(functions.print_accuracy(xref, yref))
  print("Workers updated")
  for i in range(num_workers):
    curbatch = random.choice(batches)
    xref = curbatch[0]
    yref = curbatch[1]
    results.append(functions.compute_grad(xref, yref, meanref))
  print("Grad references recieved")
  actualresult = map(lambda x: map(ray.get, x), results)
  grads = [np.asarray([gradset[i] for gradset in actualresult]) for i in range(16)] # 16 gradients, one for each variable
  gradientvalues = map(lambda x: np.mean(x, axis=0), grads) # Taking mean over all the samples
  functions.sess.run(functions.application, feed_dict=dict(zip(functions.placeholders, gradientvalues))) # Feeding the new values in
  print("End of batch {}".format(batchnum))
  batchnum += 1

services.cleanup()
