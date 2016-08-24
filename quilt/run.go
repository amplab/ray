package main

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func writeTo(path string, text string) string {
	filePath := filepath.FromSlash(path)
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY, 0777)
	if err != nil {
		fmt.Println("Could not create", filePath)
		panic(err)
	}

	defer f.Close()
	_, err = f.WriteString(text)
	if err != nil {
		fmt.Println("Error writing to", filePath)
		panic(err)
	}
	return filePath
}

func getIPs(hosts []string) []string {
	var ips []string
	for _, host := range hosts {
		ips = append(ips, waitForNetwork(host))
	}
	return ips
}

func startRay(schedulerIP string, nodeIP string) string {
	toWrite := "#!/usr/bin/python\nimport ray\n"
	start := fmt.Sprintf(`ray.services.start_node("%s", "%s", 10, `+
		`user_source_directory=None)`, schedulerIP, nodeIP)
	toWrite = toWrite + start + "\n"
	return writeTo("/root/ray/scripts/start", toWrite)
}

func startScheduler(headIP string) (string, string) {
	toWrite := "#!/usr/bin/python\nimport ray\n"
	schedulerIP := headIP + ":10001"
	start := fmt.Sprintf(`ray.services.start_scheduler("%s", cleanup=False)`,
		schedulerIP)
	toWrite = toWrite + start + "\n"
	path := writeTo("/root/ray/scripts/start_scheduler", toWrite)
	return schedulerIP, path
}

func head(host string) {
	// Get the host names of worker machines
	workerHosts := os.Getenv("WORKERHOST")
	if len(workerHosts) == 0 {
		fmt.Println("No WORKERHOST environment variable specified")
		os.Exit(1)
	}

	// XXX: getIPs will be unnecessary when hostnames are supported
	workerHostList := strings.Split(workerHosts, ",")
	workerIPList := getIPs(workerHostList)
	workerIPs := strings.Join(workerIPList, "\n")

	headIP := waitForNetwork(host)
	toWrite := headIP + "\n" + workerIPs
	writeTo("/root/ray/scripts/nodes.txt", toWrite)

	schedulerIP, schedulerPath := startScheduler(headIP)
	startPath := startRay(schedulerIP, headIP)
	toWrite = fmt.Sprintf("import ray\nray.init(node_ip_address=%s,"+
		"scheduler_address=%s\n", headIP, schedulerIP)
	writeTo("/root/ray/scripts/base.py", toWrite)

	go func() {
		fmt.Println("Starting scheduler")
		out, err := exec.Command(schedulerPath).CombinedOutput()
		if err != nil {
			fmt.Printf("Could not start scheduler: %s\n", err)
			fmt.Println(string(out))
			os.Exit(1)
		}
	}()
	time.Sleep(10 * time.Second)
	fmt.Println("Starting node")
	fmt.Println("#### START RAY INIT ####")
	fmt.Println("import ray")
	fmt.Printf(`ray.init(node_ip_address="%s", scheduler_address="%s")`, headIP,
		schedulerIP)
	fmt.Println("\n#### END RAY INIT ####")
	out, err := exec.Command(startPath).CombinedOutput()
	if err != nil {
		fmt.Printf("Could not start ray: %s\n", err)
		fmt.Println(string(out))
		os.Exit(1)
	}
}

func worker(host string) {
	fmt.Println("I'm a worker.")
	headHost := os.Getenv("HEADHOST")
	if len(headHost) == 0 {
		fmt.Println("No HEADHOST environment variable specified")
		os.Exit(1)
	}

	schedulerIP := waitForNetwork(headHost) + ":10001"
	workerIP := waitForNetwork(host)
	startPath := startRay(schedulerIP, workerIP)
	out, err := exec.Command(startPath).CombinedOutput()
	if err != nil {
		fmt.Printf("Could not start ray: %s\n", err)
		fmt.Println(string(out))
		os.Exit(1)
	}
}

func waitForNetwork(host string) string {
	fmt.Println("Waiting for", host)
	for {
		addr, err := net.LookupHost(host)
		if err != nil {
			time.Sleep(5 * time.Second)
		} else {
			fmt.Println("Network up!", addr)
			return addr[0]
		}
	}
}

func main() {
	out, err := exec.Command("/etc/init.d/ssh", "start").CombinedOutput()
	if err != nil {
		fmt.Println("Error starting ssh:", err)
		fmt.Println(string(out))
		os.Exit(1)
	}
	fmt.Println(string(out))
	host, err := os.Hostname()
	if err != nil {
		fmt.Println("No hostname found. Error:", err)
		os.Exit(1)
	}
	waitForNetwork(host)
	switch os.Args[1] {
	case "head":
		head(host)
	case "worker":
		worker(host)
	}
}
