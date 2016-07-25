One of the more time consuming tasks in machine learning is training a very deep neural network on a large dataset such as ImageNet. One of Ray's advantages is that everything except data with no available serialization can be used with no modification. Thus, this advantage allows us to use Tensorflow to build AlexNet, which is chosen due to its easy to implement architecture. Using Ray and Tensorflow, This application implements data parallel SGD on AlexNet training ImageNet. 

#Loading ImageNet  
By default, the application loads a subset of ImageNet from Amazon's s3 using the boto3 API, downloading image data from tar files and parsing label data from a text file. 

In driver.py, there are passable arguments that dictate what the s3 bucket, the key prefix for the images, and the name of the label file are:
```
parser = argparse.ArgumentParser(description="Parse information for data loading.")
parser.add_argument("--s3-bucket", default="sparknet", type=str, help="Name of the bucket that contains the image data.")
parser.add_argument("--key-prefix", default="ILSVRC2012_img_train/n015", type=str, help="Prefix for files to fetch.")
parser.add_argument("--label-file", default="train.txt", type=str, help="File containing labels")
```
After which, we setup a boto3 resource and gather the various keys for the tars to download:
```
s3 = boto3.resource("s3")
imagenet_bucket = s3.Bucket(args.s3_bucket)
objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
images = [obj.key for obj in objects.all()]
```
Once the keys are gathered, we pass them along to a remote function for parallel downloading:
```
imagenet = ray.get(imagenet.load_tarfiles_from_s3(args.s3_bucket, map(str, images), [256, 256]))
```
which returns an object reference to a list of tuples containing two object references: a batch of images in numpy array form  
and their respective filenames for use in collecting labels. 

Now we extract the batches of images:
```
X = map(lambda img: img[0], imagenet)
```
and load the file containing the label data: 
```
s5 = boto3.client("s3")
labels = s5.get_object(Bucket=args.s3_bucket, Key=args.label_file)
```
Now we have to parse the file to convert it into a dictionary for looking up the filenames efficiently and put the entire dictionary in the object store.
```
lines = labels["Body"].read().split("\n")
imagepairs = map(lambda line: line.split(" ", 2), lines)
imagepairs = ray.put(dict(map(lambda tup: (re.sub("(.+)/(.+)", r"\2", tup[0]), tup[-1]), imagepairs)))
```
To actually get the labels, we first get the actual filenames:
```
imagenames = map(lambda img: img[1], imagenet)
```
and then call a remote function `convert` to get the labels:
```
Y = map(lambda x: functions.convert(x, imagepairs), imagenames)
```
which simply looks up the filename in the previously created dictionary and converts the label to an integer.
```
@ray.remote([list, dict], [np.ndarray])
def convert(imglist, imagepairs):
  return np.asarray(map(lambda imgname:int(imagepairs[imgname]), imglist))
```

With the labels extracted, we match each array of labels with its respective batch of images:
```
batches = zip(X,Y)
```
Now, as ImageNet is in sequential form, equivalent to a fresh download from the source, we have to shuffle the images across different stored batches. Given the size of ImageNet, shuffling across the entire dataset is unfeasible. Thus, for five iterations, we permute the references to the batches:
```
newshuffle = np.random.permutation(batches)
```
and then zip the new shuffled reference and the original list to form pairs of batches to swap between:
```
batches = map(lambda tup:ray.get(functions.shufflestuples(tup)), zip(newshuffle,batches))
```
shuffling each pair in the list and then getting the resulting tuple. After the loop is done, the data is all prepared.
#AlexNet
Due to the relative size of and the limitations serializing the computation graph, each worker will possess its own copy of the network. However, to ensure that each worker is computing gradients that can be applied to all of the workers, the weights have to be synchronized over all the workers. To do this as well as initialize the network, for each corresponding weight on the network, we create an initial matrix and append that matrix to a list for updating the local network:
```
for placeholder in functions.placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
```
After that, we use Tensorflow's session evaluation to run an assignment operation to update the local network with the correct weights:
```
functions.sess.run(functions.assignment, feed_dict=dict(zip(functions.placeholders, weights)))
```
As a side note, each worker has its own Tensorflow session in which to compute its own gradient in. The main training loop starts after this, where we first create a list for the result of the compute gradient functions
```
while True:
  results = []
```
and then run the local session to get the values of the weights.
```
weights = functions.sess.run(functions.parameters)
```
For easy retrieval of the weights by the workers, we put the weights into the object store.
```
weightrefs = map(ray.put, weights)
```
Now, we launch a `update_weights` remote function for each worker:
```
for i in range(num_workers):
  functions.update_weights(*weightrefs)
```
Although the accuracy is typically printed after training is done, the accuracy is computed immediately after the weights on all the workers are updated:
```
if (batchnum % 10 == 0):
  temp = random.choice(batches)
  xref = temp[0]
  yref = temp[1]
  print ray.get(functions.print_accuracy(xref, yref))
```
The reason for this departure is because the weights are only updated at the beginning of the loop, and, as retrieving the accuracy is a remote function, the accuracy has to be computed on updated weights. If it is after the training, the local network has the correct weights, but getting the images and labels is an expensive operation that can be avoided with a remote function. 

After the accuracy is printed, SGD training takes place:
```
for i in range(num_workers):
  curbatch = random.choice(batches)
  xref = curbatch[0]
  yref = curbatch[1]
  results.append(functions.compute_grad(xref, yref))
```
We get a random batch, and pass along the reference to the images `xref` and the reference to the corresponding labels `yref` to `compute_grad`. `compute_grad` first computes a mini-batch of twenty images and labels, converting the integer labels to one hot vectors:
```
randindices = np.random.randint(0, len(X), size=[20])
subX = map(lambda ind:X[ind], randindices)
subY = np.asarray(map(lambda ind:one_hot(Y[ind]), randindices))
```
For AlexNet, the input images are 224 by 224 pixels, while the ImageNet images are 256 by 256. For additional training samples, these images are cropped randomly to the correct size for training:
```
croppedX = np.asarray(map(cropimage, subX))
```
The gradients are then computed by running the optimizer's builtin compute gradients operation stored in `compgrads`:
```
return sess.run([g for (g,v) in compgrads], feed_dict={images:croppedX, y_true:subY, dropout:0.5})
```

Once all of the computation tasks have been sent, we get the results,
```
actualresult = map(lambda x: map(ray.get, x), results)
```
group the gradients for the same variables together,
```
grads = [np.asarray([gradset[i] for gradset in actualresult]) for i in range(16)] # 16 gradients, one for each variable
```
and take the mean over each set of gradients.
```
gradientvalues = map(lambda x: np.mean(x, axis=0), grads) # Taking mean over all the samples
```
After this, we simply apply the mean gradient to the local network, training it.
```
functions.sess.run(functions.application, feed_dict=dict(zip(functions.placeholders, gradientvalues))) # Feeding the new values in
```