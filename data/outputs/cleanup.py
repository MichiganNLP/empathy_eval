"""
Module Doc String
"""

import json

with open("care_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["decoder_greedy"]
with open("care_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)


with open("cem_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["beam_search"]
with open("cem_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)


with open("human_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["human"]
with open("human_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)



with open("kemp_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["decoder_greedy"]
with open("kemp_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)


with open("mime_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["top_k"]
    query = data_obj["query"]
    data_obj["query"] = [" ".join(x) for x in query]
with open("mime_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)



with open("moel_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    data_obj["response"] = data_obj["top_k"]
    query = data_obj["query"]
    data_obj["query"] = [" ".join(x) for x in query]
with open("moel_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)


with open("seek_responses.json", "r") as file_p:
    data = json.load(file_p)
for data_obj in data:
    query = data_obj["query"]
    data_obj["query"] = [" ".join(x) for x in query]
with open("seek_responses.json", "w") as file_p:
    json.dump(data, file_p, indent=4)












def main():
    """ Driver """

if __name__ == "__main__":
    main()

