One possible application of Ray is in parallelizing reinforcement learning. In this tutorial, we will be training a neural network to play Pong using OpenAI Gym. To simplify the tutorial, I have adapted already existing code to utilize Ray from Andrej Karpathy's [excellent blog post](http://karpathy.github.io/2016/05/31/rl/), where he gives an explanation of both reinforcement learning and his code. For brevity however, I have provided a high level summary of both what reinforcement learning is for those unfamiliar in the next section. For people who are more experienced, they may skip ahead one section to a short explanation of the original code.  
##Brief Explanation of Reinforcement Learning (Optional)
The goal of reinforcement learning is to learn a policy for an agent in an environment that maximizes a given reward function. In a game, we clearly want to win, so we assign a positive reward such as 1 to a won game, and a negative reward to a lost game. In supervised learning, you have various layers which take in the input from the previous layer, performs a weighted sum on it, and then applies a non-linear transformation such as ReLU as not all inputs are directly linear to the desired outputs. The final, output layer then applies an activation function such as the softmax function to map the results of the previous layer to probabilities in the range [0,1], and then the maximum value, or the most likely outcome, is the decision the network has made. After this, we typically train the network using stochastic gradient descent or some variant.  Reinforcement learning is almost the same, but the decision is actually a policy, where the values in the final output layer are used as probabilities to flip a coin and determine what action is taken next. Furthermore, the rewards, one for a win and one for a loss, are used as labels to train the network. However, it is more likely that earlier decisions, or actions, will have a greater effect on the outcome. Thus, we want to update the rewards so that earlier rewards are worth more than later ones. After that, we use typical backpropagation to compute the gradient, and then use RMSProp to update the weights.
##Brief Explanation of the Code
As for [Andrej Karpathy's code](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), after some initial setup for the network, we enter a main loop. First, the loop preprocesses the current state of the game by marking where the paddles and ball are by 1 and everything else by 0: 
```python
  cur_x = prepro(observation)
```
and then computes the difference between this state and the previous in order to detect motion:
```python
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
```
Then, we run the network using this preprocessed data, collecting the probability of going up and what the hidden state was. Using this probability, we flip a coin to get which action we use:  
```python
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
```
After that, we simply collect the various data we need to do backpropagation, including the current state and the reward for that round (a round ends when a player scores).

Once a player has scored 21 times and has thus won the game, we discount the rewards as mentioned before:
```python
  discounted_epr = discount_rewards(epr)
  # standardize the rewards to be unit normal (helps control the gradient estimator variance)
  discounted_epr -= np.mean(discounted_epr)
  discounted_epr /= np.std(discounted_epr)
```
and then perform backpropagation on the network, storing the result into a buffer:
```python
  grad = policy_backward(eph, epdlogp)
  for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
```
We finally update the model after 10 games in order to gather some more information about the current model:
```python
  for k,v in model.iteritems():
      g = grad_buffer[k] # gradient
      rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
      model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
      grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
```
There are some areas of the code and details of reinforcement learning I did not mention, so if you want more information, I suggest reading Karpathy's blog post. Finally, you can run this code as long as you want, and you just stop it when you feel the model is playing sufficiently. Now, I will show you what I have done to speed up this training.
##Using Ray
To start, there is a canonical parallelization on computing the various gradients that can be done. Remember that we effectively compute the gradient of a game ten times before we update the model. Since we never change the model inbetween, we can simply run the ten games at the same time using different workers, and then update the model.  
To do this with Ray, first we split up the original code into three files: [functions](https://github.com/amplab/ray/tree/master/examples/rl_pong/functions.py), [workers](https://github.com/amplab/ray/tree/master/examples/rl_pong/worker.py), and then [the main file we execute](https://github.com/amplab/ray/tree/master/examples/rl_pong/driver.py).
##Ray Functions
All the functions that are going to be executed in parallel should be stored in a separate file for reasons discussed in the worker file. For simplicity however, I have moved all of the functions present in the original code to this file. For this task, the only function we need to parallelize is the one that computes the gradient. To start, I create a new function:
```python
  @ray.remote([dict], [tuple])
  def compgrad(model):
```
The `@ray.remote([dict], [tuple])` line tells Ray that this function is to be run remotely on workers, or parallelized. The arguments are to state what the types of the arguments passed in are in the first brackets, and the types of the arguments returned in the second brackets. If you want to pass more than one argument in, or more than one argument returned, you simply do [type1, type2, etc.] in the appropriate brackets. One thing to keep in mind with these argument types is that Ray only supports types that are serializable (can be constructed as a string of bytes), which for the moment are all Python primitives, dictionaries, tuples, and numpy arrays. If you have a data type you want to use with Ray, you will have to add functions that can serialize that data type.  
After the function head, most of the code is identical with the original code up to the backpropagation:
```python
  epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
  return (policy_backward(eph, epx, epdlogp, model), reward_sum)
```
reward_sum is returned together with the gradient to have a useful metric for performance. The only significant change was that, as we are only working on a single game now, gathering the data is done in a loop that finishes when the game is complete:
```python
  while not done:
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    ...
```
The last notable thing is that we create the game environment inside this file so that each worker has its own environment to work from. Something to keep in mind is that each worker has a copy of the global variables inside of these files as each worker has its own process, and we will see why later.
##Ray workers
The code in here is fairly straightforward. First, using argparse, we set the addresses for the scheduler, the object store, and the worker itself:
```python
  parser = argparse.ArgumentParser(description='Parse addresses for the worker to connect to.')
  parser.add_argument("--scheduler-address", default="127.0.0.1:10001", type=str, help="the scheduler's address")
  parser.add_argument("--objstore-address", default="127.0.0.1:20001", type=str, help="the objstore's address")
  parser.add_argument("--worker-address", default="127.0.0.1:40001", type=str, help="the worker's address")
```
Just to clarify, the scheduler manages the various workers and the tasks they are allocated, and the object store (objstore) simply stores objects in common memory for use by the various workers (as you will see in the main file). Next, when the file is actually executed, the arguments are parsed and then used by ray to connect a new worker:
```python
  args = parser.parse_args()
  ray.connect(args.scheduler_address, args.objstore_address, args.worker_address)
```
Then, we tell each worker what functions it has access to:
```python
  ray.register_module(rl_funcs)
```
This `register_module` is why all of the functions need to be in a seperate file, as your cluster that you will eventually set up needs to set up each worker before it can execute anything. Thus, if you register the same file you execute, you will execute a function that has not been set up yet. After this, we start the workers main loop, which listens for incoming tasks by the scheduler and then executes them:
```python
  worker.main_loop()
```
##Main File
As mentioned before, this is the file we execute to start the training. The code is embedded in order to gain a spatial understanding of the code. Below, there is a thorough explanation of the code. 
```python
import numpy as np
import cPickle as pickle
import gym
import ray
import ray.services as services
import rl_funcs
import os

worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "rl_worker.py")
services.start_singlenode_cluster(return_drivers=False, num_objstores=1, num_workers_per_objstore=10, worker_path=worker_path)

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

running_reward = None
batch_num = 1
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory
while True:
  modelref = ray.push(model)
  grads = []
  for i in range(batch_size):
    grads.append(rl_funcs.compgrad(modelref))
  for i in range(batch_size):
    grad = ray.pull(grads[i])
    for k in model: grad_buffer[k] += grad[0][k] # accumulate grad over batch
      running_reward = grad[1] if running_reward is None else running_reward * 0.99 + grad[1] * 0.01
      print 'Batch %d. episode reward total was %f. running mean: %f' % (batch_num, grad[1], running_reward)
  for k,v in model.iteritems():
    g = grad_buffer[k] # gradient
    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
  batch_num += 1
  if batch_num % 10 == 0: pickle.dump(model, open('save.p', 'wb'))
```
The code in this file is essentially the remnants of the original code, such as the setup for the model, summing the gradients, and updating the model. Beginning after the import statements, the file begins by finding where the file, in this case "rl_worker.py", that you want to initialize your workers with is:
```python
worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "rl_worker.py")
```
and then uses the path to start a Ray cluster in order to parallelize functions:
```python
services.start_singlenode_cluster(return_drivers=False, num_objstores=1, num_workers_per_objstore=10, worker_path=worker_path)
```
`start_singlenode_cluster` takes in how many object stores you want, how many workers to share the same object store, and the previously mentioned worker path. You should ideally have only one object store per machine to avoid unnecessary copies between object stores.  
```python
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
```
This section simply initializes the various hyperparameters we use, and resume dictates if we waqnt to continue from a previous experiment.
```python
running_reward = None
batch_num = 1
D = 80 * 80 # input dimensionality: 80x80 grid
```
Running_reward is used to measure the aggregate performance of the experiment, and batch_num simply counts which round it is. `D` is just used to initialize the model.
```python
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
```
If `resume` is set, then we simply load the model and continue the same run. Otherwise, we initialize the model as a dictionary with 'W1' being the weights for the first layer and 'W2' the weights for the second layer.
```python
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory
```
These are used for updating the weights of the model: grad_buffer for storing the total result of the gradient over the entire batch, and rmsprop_cache for storing the running average of the gradients thus far.
The next area is the main infinite loop, and the loop starts by sending the model to the object store:
```python
 modelref = ray.push(model)
```
Although we can directly pass the model into `compgrad` as it is a dictionary, it takes up less memory to place it in the object store using `ray.push`. As the scheduler has to pass the arguments by value contrary to normal Python behavior, if you are passing large objects, in this case a 200x6400 numpy array, you will take up a large chunk of memory as the scheduler has to send a copy to each worker. If the object is in the objstore, each worker already has access to it, and thus there are no problems. Now `ray.push` returns a ray.ObjRef, which is a reference to the object in objstore, for use in passing to functions, as shown in the next area.  
```python
  grads = []
  for i in range(batch_size):
    grads.append(rl_funcs.compgrad(modelref))
```
`grads` is used to simplify the process of retrieving the results of the various gradient computations. Without a list, we would need to have batch_size different variables. `Batch_size` is simply the number of games before a model update, in this case 10. When we call the ray.remote function `compgrad`, it returns an ObjRef that we then have to pull out so we append the function to `grads`. One thing to note is that although in the `@ray.remote` tag we stipulated that the argument type was a `dict`, we are actually passing an ObjRef. This is because Ray is smart enough to attach the original type to the ObjRef for type checking and to automatically pull the argument in the function body. 
```python
  for i in range(batch_size):
    grad = ray.pull(grads[i])
    for k in model: grad_buffer[k] += grad[0][k] # accumulate grad over batch
    running_reward = grad[1] if running_reward is None else running_reward * 0.99 + grad[1] * 0.01
    print 'Batch %d. episode reward total was %f. running mean: %f' % (batch_num, grad[1], running_reward)
```  
After the function eventually returns and we pull the gradient from the object store, we then add it in the gradient buffer. As mentioned before, the reward_sum returned with the function is added to running_reward with the above formula, and then printed along with the batch number and the performance of the game itself.
```python
 for k,v in model.iteritems():
    g = grad_buffer[k] # gradient
    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
```
This is where we actually update the model. As mentioned before, rmsprop_cache is the running average of the gradients squared, which is then used in the update to dampen the update. After that, we simply reset the gradient buffer.
```python
batch_num += 1
if batch_num % 10 == 0: pickle.dump(model, open('save.p', 'wb'))
```
At the end of the loop. we simply increment the batch number and every ten batches we save the model to a file.
##Takeaways
If you have a task you want to execute in parallel, you simply have to include `@ray.remote([input args], [output args])` before the function head in a seperate file. To put objects in common space for use by workers, use `ray.push`, and to retrieve objects such as the result of ray.remote functions, use `ray.pull`. Finally to setup the task, you need a worker file that has the ip addresses of the scheduler, objstore, and worker, registers every function file you want the worker to have, and starts the worker. After that, simply start a cluster with `start_singlenode_cluster` passing in the number of objstores (machines), how many workers per objstore (machine), and the file you want to initialize the workers with. Put this before the main body of your script, and you are good to go.   