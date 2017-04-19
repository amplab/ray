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


num_of_shuffles = 5 # Number of times batches are swaped with each other
num_workers = 3
batch_num = 0 # Tracker for accuracy printing and for user to see which epoch training is on
weights = [] # List to add weights of 

# Arguments to specify where the imagenet data is stored.
parser = argparse.ArgumentParser(description="Parse information for data loading.") 
parser.add_argument("--s3-bucket", required=True, type=str, help="Name of the bucket that contains the image data.")
parser.add_argument("--key-prefix", default="ILSVRC2012_img_train/n015", type=str, help="Prefix for files to fetch.")
parser.add_argument("--label-file", default="train.txt", type=str, help="File containing labels")

# Standard block for loading worker file and starting cluster.
worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "worker.py")
services.start_ray_local(num_workers=num_workers, worker_path=worker_path)

# Preparing keys for the data to be downloaded from
args = parser.parse_args()
s3 = boto3.resource("s3")
imagenet_bucket = s3.Bucket(args.s3_bucket)
objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
images = [obj.key for obj in objects.all()]
print("Keys created")

# Downloading the label file, converting each line to key:values
s3 = boto3.client("s3")
label_file = s3.get_object(Bucket=args.s3_bucket, Key=args.label_file)
name_label_pairs = label_file["Body"].read().split("\n")
image_pairs = map(lambda line: line.split(" ", 2), name_label_pairs)
image_pairs = ray.put(dict(map(lambda tup: (os.path.basename(tup[0]), tup[-1]), image_pairs)))
print("Label dictionary created")

# Downloading imagenet and computing mean image of entire set for processing batches
imagenet = ray.get(imagenet.load_tarfiles_from_s3(args.s3_bucket, map(str, images), [256, 256]))
mean_ref = functions.compute_mean_image(imagenet)
print("Imagenet downloaded and mean computed")

# Converted the parsed filenames to integer labels, creating our batches
batches = map(lambda tup: (tup[0], functions.convert(tup[-1], image_pairs)), imagenet)
print("Batches created")

# Imagenet is typically not preshuffled, so this loop does that.
if len(batches) % 2 == 0:
  batches.append(None)
print batches
for i in range(num_of_shuffles):
  batches = functions.shuffle_imagenet(batches)
  print("{}".format(ray.get(batches[0][1])))
batches = filter(lambda tup: tup != None, batches)
print("Batches shuffled")

# Initialize a matrix using a normal distribution for each weight and bias in the network 
# and assign them to the local weights. 
for placeholder in functions.placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
functions.sess.run(functions.assignment, feed_dict=dict(zip(functions.placeholders, weights)))
print("Weights passed")

while True:
  print("Start of loop")
  results = []

  # Get weights from local network and 
  weights = functions.sess.run(functions.parameters) # Retrieve weights from local network
  weight_refs = map(ray.put, weights) #Place weights into objstore
  for i in range(num_workers):
    functions.update_weights(*weight_refs) # Update the weights on each worker.
  print("Weights sent")

  #Print accuracy
  if (batch_num % 100 == 0):
    x_ref, y_ref = random.choice(batches)
    print ray.get(functions.print_accuracy(x_ref, y_ref))

  #Send the requests to compute the gradients to the workers
  for i in range(num_workers):
    x_ref,y_ref = random.choice(batches)
    results.append(functions.compute_grad(x_ref, y_ref, mean_ref))
  print("Grad references recieved")

  # Take the mean across each set of gradients and apply to the local network.
  gotten_gradients = map(lambda x: map(ray.get, x), results) # Get the actual gradients, halting the program until all are available.
  gradients = [np.asarray([grad_set[i] for grad_set in gotten_gradients]) for i in range(16)] # 16 gradients, one for each variable
  gradient_mean = map(lambda x: np.mean(x, axis=0), gradients) # Taking mean over all the samples
  functions.sess.run(functions.application, feed_dict=dict(zip(functions.placeholders, gradient_mean))) # Feeding the new values in
  print("End of batch {}".format(batch_num))
  batch_num += 1

services.cleanup()
