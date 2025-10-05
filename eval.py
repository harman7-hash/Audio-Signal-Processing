import yaml

with open("./results.yaml") as stream:
    param = yaml.safe_load(stream)
param["results"].append("haram")


with open('./results.yaml', 'w') as file:
    yaml.dump(param, file, default_flow_style=False)