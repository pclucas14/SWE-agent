# retrieve code graph scripts

import os
import pickle
import sys
import json

def main(repo_name, func_name):
    try:
        with open(f'/root/tools/repograph/{repo_name}_graph.pkl', 'rb') as f:
            G = pickle.load(f)
    except:
        print("No graph file found")
        return []
    try:
        with open(f'/root/tools/repograph/{repo_name}_tags.json', 'r') as f:
            tags = f.readlines()
    except:
        print("No tags file found")
        return []
    tags = [json.loads(tag) for tag in tags]

    # try:
    successors = list(G.successors(func_name))
    predecessors = list(G.predecessors(func_name))
    tags2names = {tag['name']: tag for tag in tags}
    returned_files = []
    for item in successors+[func_name]+predecessors:
        # if 'test' in tags2names[item]['fname']:
        #     continue
        returned_files.append({
            "fname": tags2names[item]['fname'],
            'line': tags2names[item]['line'],
            'name': tags2names[item]['name'],
            'kind': tags2names[item]['kind'],
            'category': tags2names[item]['category'],
            'info': tags2names[item]['info'],
        })
    print(f"Done searching from repo: {returned_files}")
    # except:
    #     print("None")
    return returned_files

# repograph/retrieve_graph.py
def run(repo_name, search_term):
    print(f"Searching for {search_term} in {repo_name}")
    # your actual logic here
    main(repo_name, search_term)


# if __name__ == '__main__':
#     func_name = sys.argv[1]
