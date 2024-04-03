from sys import argv
import subprocess
import os 
import json
import numpy as np

def load_graph(graph_file:str) -> np.ndarray:
    f = open(graph_file)
    content = f.read()
    f.close()
    lines = content.splitlines()
    lines = lines[1:]
    e = lines[0].split(",")
    out = np.zeros(shape=(len(e),len(e)))
    for i, line in enumerate(lines):
        vals = line.split(",")
        for j, val in enumerate(vals):
            if val != '':
                out[i, j] = int(val)
    return out

def check_negative_cycles(graph:np.ndarray):
    num_vertices = graph.shape[0]
    distances = [float('inf')] * num_vertices
    distances[0] = 0

    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v in range(num_vertices):
                if graph[u][v] != 0 and distances[u] + graph[u][v] < distances[v]:
                    distances[v] = distances[u] + graph[u][v]

    for u in range(num_vertices):
        for v in range(num_vertices):
            if graph[u][v] != 0 and distances[u] + graph[u][v] < distances[v]:
                return True

    return False

def check_distances(distances:list[int], predecessors:list[int], graph:np.ndarray):
    source = 0
    for i, dist in enumerate(distances):
        if dist == 0:
            source += 1
            continue
        computed_dist = 0
        prev = predecessors[i]
        next = i
        while prev != -1:
            computed_dist += graph[prev, next]
            next = prev
            prev = predecessors[prev]
        if dist != computed_dist:
            return False
    return True
    
def load(param:str)-> list[dict]:
    if os.path.isfile(param):
        with open(param) as f:
            return [json.loads(f.read())]

    if os.path.isdir(param):
        files = [os.path.join(param,f) for f in os.listdir(param) if os.path.isfile(os.path.join(param,f))]
        elements = []
        for file in files:
            with open(file) as f:
                elements.append(json.loads(f.read()))

        return elements
    return []

def compute(param:str, compute_range:str) -> list[dict]:
    params = []
    if os.path.isfile(param):
        params =  [param]

    if os.path.isdir(param):
        params = [os.path.join(param,f) for f in os.listdir(param) if os.path.isfile(os.path.join(param,f))]
    
    process_range = range(1)
    if "-" in compute_range:
        extremes = compute_range.split("-")
        process_range = range(int(extremes[0]), int(extremes[1]) + 1)
    else:
        process_range = [int(compute_range)]

    results = []
    for param in params:
        for num_threads in process_range:
            command = ["output/bellman-ford", param]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS=str(num_threads)))
            results.append(json.loads(result.stdout))

    return results

def organise(runs: list[dict]) -> dict:
    out = {}
    for run in runs:
        input_file = run["input_file"]
        if not input_file in out:
            out[input_file] = []
        out[input_file].append(run)
    return out

def check(runs:dict) -> dict:
    checked_runs = runs.copy()
    for instance in checked_runs.keys():
        instance_runs = checked_runs[instance]
        graph = load_graph(instance)
        has_negative_cycle = check_negative_cycles(graph)
        for run in instance_runs:
            run["correct"] = True
            neg = bool(run["negative_cycles"])
            if not neg == has_negative_cycle:
                run["correct"] = False
                print("false negative cycle value")
            if not check_distances(run["distances"], run["predecessors"], graph):
                print("cost values not correct")
                run["correct"] = False
    return checked_runs

def load_args(argv:list[str]) -> dict[str,str]:
    args = {
        "verbose":False,
        "mode": argv[1]
    }
    if argv[1] == "load":
        args["params"] = argv[2]
        argv = argv[3:]
    elif argv[1] == "compute":
        args["params"] = argv[2]
        args["range"] = argv[3]
        argv = argv[4:]
    else:
        raise Exception(f"Unrecognised mode {argv[1]}")
    for arg in argv:
        if "--verbose" == arg:
            args["verbose"] = True
        elif "--save" in arg:
            args["save"] = arg.replace("--save=","")
        elif "--makePlots" in arg:
            args["plots"] = arg.replace("--makePlots=", "")
    
    return args

def main():
    if len(argv) < 2:
        print(f"Wrong usage.Use {argv[0]} --help if needed")
        return
    if argv[1] == "--help":
        print(f"""Usage: {argv[0]} mode [flags]
Modes:
    load    load a file/folder.
        Usage: {argv[0]} load file-folder [flags]
    compute call the program output/bellman-ford on a given instance file/folder and with a given range of cores.
        Usage: {argv[0]} compute file/folder range(e.g. 1-8 or 6) [flags]
Flags:
    --verbose   print the output to the stdout
    --save      save the output to a given file. Usage: {argv[0]} mode --save=fileName
    --makePlots create plots of the analised instances and save them to a given folder. Usage: {argv[0]} mode --makePlots=folderName
""")
        return
    args = load_args(argv)
    mode = args["mode"]
    modes = {"load":lambda **args: load(args["params"]), "compute":lambda **args:compute(args["params"], args["range"])}
    runs = modes[mode](**args)
    runs = organise(runs)
    runs = check(runs)
    out = [(run["cores"], run["execution_time"], run["correct"]) for run in runs[args["params"]]]
    for o in out:
        print(o)

if __name__ == "__main__":
    main()
