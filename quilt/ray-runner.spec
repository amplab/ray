(import "ray") // Import ray.spec.

// We will have one worker node.
(define nWorker 1)

// Create Ray containers, using the New function in ray.spec.
(let ((rayContainer (ray.New "ray" nWorker)))
    (ray.Job rayContainer "driver.py"))

// The namespace of the cluster that will be booted.
(define Namespace "vivian-ray-1")

// Defines the set of addresses that are allowed to access Quilt VMs.
(define AdminACL (list "local"))

// Specify the configuration of each machine. Each machine will boot on Amazon,
// with 32 GB of disk space. If you have set up SSH keys for your GitHub
// account, replace YOUR_GITHUB_USERNAME with your GitHub username. Otherwise,
// replace line 24 with (sshKey "YOUR_SSH_KEY").
(let ((cfg (list (provider "Amazon")
                 (region "us-west-1")
                 (size "m4.large")
                 (diskSize 32)
                 (githubKey "secant"))))
     (makeList 1 (machine (role "Master") cfg))
     (makeList nWorker (machine (role "Worker") cfg)))
