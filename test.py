import os, json

import unstructured_client
from unstructured_client.models import operations, shared

client = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    #  server_url=os.getenv("UNSTRUCTURED_API_URL"),
    server_url="http://10.69.1.169:9025"
)

filename = "/app/docs/python/3.13/library/array.html"

req = operations.PartitionRequest(
    partition_parameters=shared.PartitionParameters(
        files=shared.Files(
            content=open(filename, "rb"),
            file_name=filename,
        ),
        strategy=shared.Strategy.HI_RES,
        languages=['eng'],
        split_pdf_page=True,            # If True, splits the PDF file into smaller chunks of pages.
        split_pdf_allow_failed=True,    # If True, the partitioning continues even if some pages fail.
        split_pdf_concurrency_level=15  # Set the number of concurrent request to the maximum value: 15.
    ),
)

try:
    res = client.general.partition(request=req)
    element_dicts = [element for element in res.elements]

    # Print the processed data's first element only.
    print(element_dicts[0])

    # Write the processed data to a local file.
    json_elements = json.dumps(element_dicts, indent=2)

    with open("PATH_TO_OUTPUT_FILE", "w") as file:
        file.write(json_elements)
except Exception as e:
    print(e)