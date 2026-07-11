import json

with open("output/schema.json") as f:

    schema = json.load(f)

metadata = {}

for db in schema:

    metadata[db] = {}

    for table in schema[db]:

        metadata[db][table] = {}

        for col in schema[db][table]:

            name = col["column"]

            metadata[db][table][name] = (

                f"{name} is a column "

                f"from {table} table."

            )

with open(

    "output/metadata.json",

    "w"

) as f:

    json.dump(metadata, f, indent=4)

print("Metadata Generated")