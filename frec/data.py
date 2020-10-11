import json
import codecs


def load_metadata_from_json(path_to_json):
    json_data = []

    with codecs.open(path_to_json, 'rU', 'utf-8') as js_file:
        for line in js_file:
            json_data.append(json.loads(line))

    print(f"{len(json_data)} image found")

    return json_data
