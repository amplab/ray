import numpy as np
import ray
import os
import argparse
import boto3

import alexnet

# Arguments to specify where the imagenet data is stored.
parser = argparse.ArgumentParser(description="Parse information for data loading.")
parser.add_argument("--s3-bucket", required=True, type=str, help="Name of the bucket that contains the image data.")
parser.add_argument("--key-prefix", default="ILSVRC2012_img_train/n015", type=str, help="Prefix for files to fetch.")
parser.add_argument("--label-file", default="train.txt", type=str, help="File containing labels")

if __name__ == "__main__":
  args = parser.parse_args()
  num_workers = 4
  ray.services.start_ray_local(num_workers=num_workers)

  # Note we do not do sess.run(tf.initialize_all_variables()) because that would
  # result in a different initialization on each worker. Instead, we initialize
  # the weights on the driver and load the weights on the workers every time we
  # compute a gradient.
  ray.reusables.net_vars = ray.Reusable(alexnet.net_initialization, alexnet.net_reinitialization)

  # Prepare keys for downloading the data.
  s3_resource = boto3.resource("s3")
  imagenet_bucket = s3_resource.Bucket(args.s3_bucket)
  objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
  image_tar_files = [str(obj.key) for obj in objects.all()]
  print "Images will be downloaded from {} files.".format(len(image_tar_files))

  # Downloading the label file, and create a dictionary mapping the filenames of
  # the images to their labels.
  s3_client = boto3.client("s3")
  label_file = s3_client.get_object(Bucket=args.s3_bucket, Key=args.label_file)
  filename_label_str = label_file["Body"].read().strip().split("\n")
  filename_label_pairs = [line.split(" ") for line in filename_label_str]
  filename_label_dict = dict([(os.path.basename(name), label) for name, label in filename_label_pairs])
  filename_label_dict_ref = ray.put(filename_label_dict)
  print "Labels extracted"

  # Download the imagenet dataset.
  imagenet_data = alexnet.load_tarfiles_from_s3(args.s3_bucket, image_tar_files, [256, 256])

  # Convert the parsed filenames to integer labels and create batches.
  batches = [(images, alexnet.filenames_to_labels(filenames, filename_label_dict_ref)) for images, filenames in imagenet_data]

  # Compute the mean image.
  mean_ref = alexnet.compute_mean_image([images for images, labels in batches])

  # The data does not start out shuffled. Images of the same class all appear
  # together, so we shuffle it ourselves here. Each shuffle pairs up the batches
  # and swaps data within a pair.
  num_shuffles = 5
  for i in range(num_shuffles):
    batches = alexnet.shuffle(batches)

  _, sess, application, _, _, _, _, placeholders, parameters, assignment, init_all_variables = ray.reusables.net_vars
  # Initialize the network and optimizer weights. This is only run once on the
  # driver. We initialize the weights manually on the workers.
  sess.run(init_all_variables)
  print "Initialized network weights."

  iteration = 0
  while True:
    # Extract weights from the local copy of the network.
    weights = sess.run(parameters)
    # Put weights in the object store.
    weights_ref = ray.put(weights)

    # Compute the accuracy on a random training batch.
    x_ref, y_ref = batches[np.random.randint(len(batches))]
    accuracy = alexnet.compute_accuracy(x_ref, y_ref, weights_ref)

    # Launch tasks in parallel to compute the gradients for some batches.
    gradient_refs = []
    for i in range(num_workers - 1):
      # Choose a random batch and use it to compute the gradient of the loss.
      x_ref, y_ref = batches[np.random.randint(len(batches))]
      gradient_refs.append(alexnet.compute_grad(x_ref, y_ref, mean_ref, weights_ref))

    # Print the accuracy on a random training batch.
    print "Iteration {}: accuracy = {:.3}%".format(iteration, 100 * ray.get(accuracy))

    # Fetch the gradients. This blocks until the gradients have been computed.
    gradient_sets = [ray.get(ref) for ref in gradient_refs]
    # Average the gradients over all of the tasks.
    mean_gradients = [np.mean([gradient_set[i] for gradient_set in gradient_sets], axis=0) for i in range(len(weights))]
    # Use the gradients to update the network.
    sess.run(application, feed_dict=dict(zip(placeholders, mean_gradients)))

    iteration += 1
