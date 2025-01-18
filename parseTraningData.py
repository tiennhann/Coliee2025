from turtle import pd
import xml.etree.ElementTree as ET
import sys

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
        record_data = {
            'id': query_id,
            'query': query,
            'paragraphs': paragraphs 
        }

        records.append(record_data)

    return records

def records_to_dataframe(records):
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    parsed_records = parse_xml_file(sys.argv[1])

    # Print results (or process further)
    for r in parsed_records:
        print(r)
