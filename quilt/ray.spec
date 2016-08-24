(import "github.com/NetSys/quilt/specs/stdlib/strings")

// The Docker image that will be pulled.
(define image "vfang/ray")

// A convenience function for joining a list of hostnames with commas.
(define (commaSepHosts labels)
    (strings.Join (map labelHost labels) ","))

// Create the head node.
(define (createHead prefix)
    (let ((labelNames (strings.Range (sprintf "%s-ms" prefix) 1))
          (rayHead (makeList 1 (docker image "run" "head"))))
        (map label labelNames rayHead)))

// Create the worker node(s).
(define (createWorkers prefix n)
    (let ((labelNames (strings.Range (sprintf "%s-wk" prefix) n))
          (rayWorkers (makeList n (docker image "run" "worker"))))
        (map label labelNames rayWorkers)))

// Map the hostnames to each head and worker node. Also opens the SSH port.
(define (link head workers)
    (setEnv head "HEADHOST" (commaSepHosts head))
    (setEnv head "WORKERHOST" (commaSepHosts workers))
    (setEnv workers "HEADHOST" (commaSepHosts head))
    (connect 22 head workers)
    (connect 22 workers head)
    (connect (list 100 65535) head workers)
    (connect (list 100 65535) workers head))

// This is the function that other specs will call. It creates one head node
// and nWorker worker nodes. Returns of a map of the head node and worker
// nodes.
(define (New prefix nWorker)
    (let ((workers (createWorkers prefix nWorker))
          (head (createHead prefix)))
         (link head workers)
         (hmap ("head" head)
               ("worker" workers))))

// Will map the environment variable JOB to <command> to the head in the
// <rayMap> (the map returned by New)
(define (Job rayMap command)
    (setEnv (hmapGet rayMap "head") "JOB" command))
