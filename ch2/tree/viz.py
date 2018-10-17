import graphviz
# On WSL, the following doesn't work cuz no diplay

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
