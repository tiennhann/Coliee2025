import glob
import os
import random
from turtle import pd
import xml.etree.ElementTree as ET
import sys

import xml.etree.ElementTree as ET
from dspy.datasets  import Dataset, HotPotQA

def parse_xml_file(file_path):
    # 1. Parse the XML file into an ElementTree object
    tree = ET.parse(file_path)

    # 2. Get the root of the XML (in our example, 'dataset')
    root = tree.getroot()

    # 3. Create a list to store our parsed data
    records = []

    # 4. Loop through each record in the dataset
    for pair in root.findall('pair'):
        query_id = pair.attrib["id"]
        query = pair.find('t2')
        query = query.text if query is not None else None
        paragraphs  = pair.find('t1')
        paragraphs  = paragraphs.text if query is not None else None


        query =  query.strip()
        paragraphs = paragraphs.strip()
        query_id = query_id.strip()

        # Store this record in a dictionary (or any structure of your choice)
        if id is None or query is None or paragraphs is None:
            continue
        record_data = {
            'id': query_id,
            'query': query,
            'answer': paragraphs,
        }

        records.append(record_data)

    return records

def parse_all_xml_files(folder_path):
    xml_files = glob.glob(os.path.join(folder_path, '*.xml'))
    data = []
    for file in xml_files:
        data.extend(parse_xml_file(file))
    return data

class CollieDataset(Dataset):
    def __init__(
        self,
        *args,
        train_folder,
        test_folder=None,
        dev_folder=None,
        split_percent=75,
        **kwargs,
    ):
       
        super().__init__(*args, **kwargs)

        examples = parse_all_xml_files(train_folder)    
        rng = random.Random(0)
        rng.shuffle(examples)

        self._train = examples[: len(examples) * split_percent // 100]
        if test_folder is None:
            self._test = examples[len(examples) * split_percent // 100 :]
        else:
            self._test = parse_all_xml_files(test_folder)
        
        if dev_folder is not None:
            self._dev = parse_all_xml_files(dev_folder)

if __name__ == "__main__":
    parsed_records = parse_xml_file(sys.argv[1])

    # Print results (or process further)
    for r in parsed_records:
        print(r)
