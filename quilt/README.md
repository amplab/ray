# Running Ray with Quilt

Quilt is a way to deploy and network containers, meaning that Quilt will boot
machines and set up containers on supported cloud providers (AWS, GCE, Azure)
for you. Learn more about this project [here](http://quilt.io).

## Introduction
Quilt uses a domain specific language, Stitch (files that end with `.spec`, to 
specify distributed applications, independent of the specific infrastructure
they run on. This folder contains [`ray.spec`](ray.spec) and 
[`ray-runner.spec`](ray-runner.spec). `ray.spec` provides abstractions for
creating ray containers - the function `ray.New` creates `nWorker` worker nodes,
and one head node, all prefixed with `prefix`.

`ray-runner.spec` defines the infrastructure When you run `ray-runner.spec`, Quilt will

## Step 1: Installing Quilt

### Install Go
Quilt supports Go version 1.5 or later.

Install Go using your package manager or on the 
[Golang website](https://golang.org/doc/install).

#### Set up GOPATH
We recommend reading the overview to Go workplaces 
[here](https://golang.org/doc/code.html).

Before installing Quilt, you'll need to set up your `GOPATH`. Assuming the root
of your Go workspace will be `~/gowork`, execute the following `export`
commands in your terminal to set up your `GOPATH`:

```bash
export GOPATH=~/gowork
export PATH=$PATH:$GOPATH/bin
```

It would be a good idea to add these commands to your `.bashrc` (or
`.bash_profile` on OS X) so that they do not have to be run again.

### Download and Install Quilt
Clone the repository into your Go workplace: `go get github.com/NetSys/quilt`.

This command also automatically installs Quilt. If the installation was
successful, then the `quilt` command should execute in your shell.

### QUILT_PATH
Your `QUILT_PATH` will be where Quilt looks for imported specs, and where
specs you download with `quilt get` get placed. You can set this to be anywhere,
but by default, your `QUILT_PATH` is `~/.quilt`. To set a custom `QUILT_PATH`,
follow the instructions
[here](https://github.com/NetSys/quilt/blob/master/docs/Stitch.md#quilt_path).

### Download Imported Specs
If you look at [`ray.spec`](ray.spec), the first line imports 
`github.com/NetSys/quilt/specs/stdlib/strings`. To resolve these imports, simply
execute `quilt get github.com/amplab/ray`. This will clone the ray repository
into your `QUILT_PATH`, and will also download imports of the specs that were
downloaded.

## Step 2: Configure `ray-runner.spec` & Cloud Provider
In [`ray-runner.spec`](ray-runner.spec), there are two things that will have to
be filled in.

### Configure SSH Authentication
Quilt-managed Machines use public key authentication to control SSH access.
To start using the Ray cluster, we will need to access the Master VM.

If you would like to use `githubKey` authentication, go to line 24 of 
[`ray-runner.spec`](ray-runner.spec) and fill in  `(define githubKey
"YOUR_GITHUB_USERNAME")` appropriately. Instructions on adding a key to your
GitHub account can be found
[here](https://help.github.com/articles/generating-an-ssh-key/).

If you would ike to use an SSH key instead, replace `(define githubKey
"YOUR_GITHUB_USERNAME")` with `(define sshkey "YOUR_PUBLIC_KEY")`.

`"YOUR_PUBLIC_KEY"` should be a string with the contents of your
`~/.ssh/id_rsa.pub`, e.g.: `(define sshkey "ssh-rsa
AAAAB3NzaC1yc2EAAAADAQABAAABAQ example@example.com")`.

### Choose a Namespace
Running two Quilt instances with the same Namespace is not supported.
If you are sharing a computing cluster with others, it would be a good idea to
change `(define Namespace "CHANGE_ME")` on line 11 to a different name.

### Configure a Cloud Provider
Below we discuss how to setup Quilt for Amazon EC2. Other providers are
supported as well, including Microsoft Azure and Google Compute Engine. 
Since Quilt deploys systems consistently across providers, the details of the
rest of this document will apply no matter what provider you choose.

For Amazon EC2, you'll first need to create an account with [Amazon Web
Services](https://aws.amazon.com/ec2/) and then find your
[access credentials](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-set-up.html#cli-signup).
That done, you simply need to populate the file `~/.aws/credentials`, with your
Amazon credentials:

```
[default]
aws_access_key_id = <YOUR_ID>
aws_secret_access_key = <YOUR_SECRET_KEY>
```

## Step 3: Run `ray-runner.spec`

### Start the Quilt daemon
Execute `quilt daemon` in one terminal session. This will start the quilt
process (and hang). Do not close this session. Open a new terminal session
(<kbd>Ctrl</kbd> + <kbd>T</kbd> or <kbd>&#8984;</kbd> + <kbd>T</kbd> on OS X).

### Run `ray-runner.spec`
We suggest you read through [`ray.spec`](ray.spec) and
[`ray-runner.spec`](ray-runner.spec) to get a feel of how the infrastructure is
defined.

While in the `path/to/amplab/ray/quilt/` directory, execute `quilt run
ray-runner.spec`. Quilt will set up `nWorker + 1` VMs. You will see logs pop up
on the terminal session where you executed `quilt daemon`.

### Accessing the VM
It will take a while for the VMs to boot up, for Quilt to configure the network,
and for Docker containers to be initialized. When the logs in the Quilt daemon
display machines with IP addresess and say `Connected`:

```
quilt daemon
INFO [Aug 16 15:04:45.104] db.Machine:
    Machine-2{Master, Amazon us-west-1 m4.large, Disk=32GB}
    Machine-3{Worker, Amazon us-west-1 m4.large, Disk=32GB}
INFO [Aug 16 15:04:45.105] db.Cluster:
    Cluster-1{vivian-ray-1, ACL: [136.152.142.34/32]}
INFO [Aug 16 15:04:56.364] Attempt to boot machines.                count=2
INFO [Aug 16 15:05:59.867] Successfully booted machines.
INFO [Aug 16 15:06:10.772] db.Machine:
    Machine-2{Master, Amazon us-west-1 m4.large, sir-042pcgha, PublicIP=54.193.20.57, PrivateIP=172.31.11.175, Disk=32GB}
    Machine-3{Worker, Amazon us-west-1 m4.large, sir-042r83pp, PublicIP=54.183.84.139, PrivateIP=172.31.9.27, Disk=32GB}
INFO [Aug 16 15:08:15.114] db.Machine:
    Machine-2{Master, Amazon us-west-1 m4.large, sir-042pcgha, PublicIP=54.193.20.57, PrivateIP=172.31.11.175, Disk=32GB}
    Machine-3{Worker, Amazon us-west-1 m4.large, sir-042r83pp, PublicIP=54.183.84.139, PrivateIP=172.31.9.27, Disk=32GB, Connected}
INFO [Aug 16 15:09:45.116] db.Machine:
    Machine-2{Master, Amazon us-west-1 m4.large, sir-042pcgha, PublicIP=54.193.20.57, PrivateIP=172.31.11.175, Disk=32GB, Connected}
    Machine-3{Worker, Amazon us-west-1 m4.large, sir-042r83pp, PublicIP=54.183.84.139, PrivateIP=172.31.9.27, Disk=32GB, Connected}
```

You can ssh into a machine by executing `quilt ssh <MACHINE_ID>` in the same
terminal session you executed. To view available machines, execute `quilt
machines`, which will give output like this:

```
quilt machines
Machine-2{Master, Amazon us-west-1 m4.large, sir-042pcgha, PublicIP=54.193.20.57, PrivateIP=172.31.11.175, Disk=32GB, Connected}
Machine-3{Worker, Amazon us-west-1 m4.large, sir-042r83pp, PublicIP=54.183.84.139, PrivateIP=172.31.9.27, Disk=32GB, Connected}
```

The `MACHINE_ID` is the number after `Machine-` in the output of `quilt
machines`. For example, to ssh into in the Master VM, execute `quilt ssh 2`.

### Finding Ray
Once you are in a ssh session with the Master VM, execute `swarm ps`. This will
list all of the containers on the network. Look for the containers with the ray
image (`44278ea4ffe2` and `e23843516694` in the example below).

```
quilt@ip-172-31-11-175:~$ swarm ps
CONTAINER ID        IMAGE                        COMMAND                  CREATED             STATUS              PORTS               NAMES
93cac6c0b38e        vfang/ray                    "run worker"             2 minutes ago       Up About a minute                       ip-172-31-9-27/modest_bardeen
80c7a93d1d50        vfang/ray                    "run head"               2 minutes ago       Up About a minute                       ip-172-31-9-27/amazing_ritchie
0a2739052a96        quilt/ovs                    "run ovn-controller"     4 minutes ago       Up 4 minutes                            ip-172-31-9-27/ovn-controller
3421fadfb450        quay.io/coreos/etcd:v3.0.2   "/usr/local/bin/etcd "   6 minutes ago       Up 6 minutes                            ip-172-31-9-27/etcd
245d44571e20        quilt/ovs                    "run ovs-vswitchd"       6 minutes ago       Up 6 minutes                            ip-172-31-9-27/ovs-vswitchd
93e392584eb3        quilt/ovs                    "run ovsdb-server"       6 minutes ago       Up 6 minutes                            ip-172-31-9-27/ovsdb-server
b38af80d1dfc        quilt/quilt:latest           "/quilt minion"          6 minutes ago       Up 6 minutes                            ip-172-31-9-27/minion
```

If they have not shown up, wait a few more minutes (see the time differences
in creation between the minion container and the ray containers).

When they show up, you will want to access the head node, so look for the
container ID of the container with the command `run head`. In the example above,
the container ID will be `80c7a93d1d50`.

### Playing With Ray
The starter code needed to write a ray script looks like this:
```python
import ray
ray.init(node_ip_address="<HEAD_IP_ADDRESS>", scheduler_address="<HEAD_IP_ADDRESS>:10001")
```

This is printed to the logs of the container, which can be accessed by
executing the command `swarm logs <HEAD_CONTAINER_ID>`. Following the example
above, we would execute `swarm logs 80c7a93d1d50`. This gives the output:

```
quilt@ip-172-31-4-200:~$ swarm logs b075566e3559
   Starting OpenBSD Secure Shell server sshd
      ...done.

Waiting for 80c7a93d1d50
Network up! [10.0.9.118]
Waiting for ray-wk-0.q
Network up! [10.0.155.142]
Waiting for 80c7a93d1d50
Network up! [10.0.9.118]
Starting scheduler
Starting node
#### START RAY INIT ####
import ray
ray.init(node_ip_address="10.0.9.118", scheduler_address="10.0.9.118:10001")
#### END RAY INIT ####
```

Using this code, we can run the provided [`driver.py`](scripts/driver.py) to
calculate pi on the cluster. This file is located in `~/ray/scripts`. 

Uncomment line 5, and fill in the `node_ip_address`
and `scheduler_address` (vim is installed on the container). You can also make
your own script, or open the python interpreter.

This is the result of executing `python ~/root/scripts/driver.py`
```
root@80c7a93d1d50:~/ray/scripts# python driver.py
I0816 22:21:14.206347342     146 ev_epoll_linux.c:84]        epoll engine will be using signal: 36
D0816 22:21:14.206399619     146 ev_posix.c:106]             Using polling engine: epoll
Pi is approximately 3.11600000858.
Successfully shut down Ray.
```

## Extra: Customize the Ray Container
If you would like to load your own scripts into the container, you can add
files to the [`scripts`](scripts/) folder, which will get copied over to
`~/ray/scripts` when you rebuild the Dockerfile.

When you [rebuild the Dockerfile](https://docs.docker.com/engine/tutorials/dockerimages/)
(or have a custom image you would like to use), make sure to edit line 4 of the
`ray.spec` (in your `$QUILT_PATH`, which is by default `~/.quilt`) file to
point to your own image on Dockerhub.

```
(define image "<USERNAME>/<IMAGE>")
```
